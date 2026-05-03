import torch
import torch.nn as nn
import torch.nn.functional as F
import math




def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 10000.0) -> torch.Tensor:
    
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, got {embed_dim}")
        
    device = pos.device
    # 生成频率项
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)
    omega /= (embed_dim / 2.0)
    omega = 1.0 / (omega_0 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 10000.0) -> torch.Tensor:
   
    # pos_grid: [W, H, 2] (注意 create_uv_grid 的输出可能是 WxH)
    # 我们先统一把输入视为 [*dims, 2]
    *spatial_dims, grid_dim = pos_grid.shape
    assert grid_dim == 2, "pos_grid last dim must be 2"
    
    # 展平处理
    pos_flat = pos_grid.reshape(-1, grid_dim)  # (N, 2)

    # 核心：X 和 Y 各占一半维度
    half_dim = embed_dim // 2
    
    # 处理 X 坐标 (pos_flat[:, 0]) -> [N, D/2]
    emb_x = make_sincos_pos_embed(half_dim, pos_flat[:, 0], omega_0=omega_0)
    
    # 处理 Y 坐标 (pos_flat[:, 1]) -> [N, D/2]
    emb_y = make_sincos_pos_embed(half_dim, pos_flat[:, 1], omega_0=omega_0)

    # 拼接 -> [N, D]
    emb = torch.cat([emb_x, emb_y], dim=-1)

    # 还原空间维度
    return emb.view(*spatial_dims, embed_dim)

def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = torch.float32, device: torch.device = None
) -> torch.Tensor:
    
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # 计算归一化的 span
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # 生成边界
    # 注意：这里用 (width-1) 和 (height-1) 确保覆盖边界
    left_x = -span_x 
    right_x = span_x 
    top_y = -span_y 
    bottom_y = span_y 

    # 生成坐标 linspace
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # 生成网格 (indexing='xy' -> x是宽, y是高)
    # uu: [W, H], vv: [W, H]
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    
    # 堆叠 -> [W, H, 2]
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid

class AlignProjector(nn.Module):
   
    def __init__(
        self, 
        vit_dim: int, 
        dpt_dim: int, 
        vit_resolution: tuple = (16, 16), # ViT 输入分辨率 (256 tokens)
        dpt_resolution: tuple = (36, 36), # DPT 目标分辨率 (1296 tokens)
        use_norm: bool = True
    ):
        super().__init__()
        self.vit_h, self.vit_w = vit_resolution
        self.dpt_h, self.dpt_w = dpt_resolution
        self.vit_dim = vit_dim
        
        # LayerNorm (对齐前处理)
        self.norm = nn.LayerNorm(vit_dim) if use_norm else nn.Identity()
        
        # 投影层: 放在最后，负责将 vit_dim 映射到 dpt_dim
        # 输入维度是 vit_dim，因为我们是先插值 (不改变维度) 后投影
        self.net = nn.Sequential(
            nn.Linear(vit_dim, dpt_dim*2),
            nn.GELU(),
            nn.Linear(dpt_dim*2, dpt_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, vit_feats):
        """
        Args:
            vit_feats: [B*N, 256, vit_dim]
        Returns:
            aligned_feats: [B*N, 1296, dpt_dim] (Ready for DPT Decoder / Loss)
        """
        # [B*N, 256, vit_dim]
        b, n_tokens, dim = vit_feats.shape
        
        # 1. 还原 2D 结构
        # [B*N, vit_dim, 16, 16] (Channel First for Interpolate/PE)
        x = vit_feats.view(b, self.vit_h, self.vit_w, dim).permute(0, 3, 1, 2)
        
        # 2. 加位置编码 (Add PE)
        grid = create_uv_grid(width=self.vit_w, height=self.vit_h, dtype=x.dtype, device=x.device)
        # 转置为 [H, W, 2] 以匹配特征图布局
        grid = grid.permute(1, 0, 2) 
        # 生成 PE [H, W, 896]
        pe = position_grid_to_embed(grid, embed_dim=dim)
        # 调整维度 [H, W, 896] -> [1, 896, H, W]
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        # 相加
        x = x + pe


        # 3. 插值/上采样 (Interpolate)
        # [B*N, vit_dim, 16, 16] -> [B*N, vit_dim, 36, 36]
        x = F.interpolate(
            x, 
            size=(self.dpt_h, self.dpt_w), 
            mode='bilinear', 
            align_corners=True
        )
        
        # 4. 展平 (Flatten)
        # [B*N, vit_dim, 36, 36] -> [B*N, 1296, vit_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # 5. 投影 (Project) - 最后一步
        # [B*N, 1296, vit_dim] -> [B*N, 1296, dpt_dim]
        x = self.norm(x)
        out = self.net(x)
        
        return out

class SpatialForcingLoss(nn.Module):
    def __init__(self, vit_dim, dpt_dim, vit_res=(16, 16), dpt_res=(36, 36)):
        super().__init__()
        
        self.projector = AlignProjector(
            vit_dim=vit_dim, 
            dpt_dim=dpt_dim, 
            vit_resolution=vit_res,
            dpt_resolution=dpt_res,
            use_norm=True
        )
        
        # reduction='mean' 会对所有元素求平均，我们将在过滤后的有效元素上做
        self.loss_fn = nn.CosineEmbeddingLoss(reduction='mean')
        
    def forward(self, vit_feats, dpt_feats, mask):
        """
        Args:
            vit_feats: [B, 3, 256, vit_dim]  (Student Input)
            dpt_feats: [B, 3, 1296, dpt_dim] (Target)
            mask:      [B, 3] (Boolean Tensor, True=Valid, False=Dummy)
            
        Returns:
            loss: scalar
        """
        # 1. 展平 Batch 和 View 维度
        # [B*3, 256, vit_dim]
        #print(f"\n[SF_DEBUG] Input Shapes:")
        #print(f"  vit_feats: {vit_feats.shape}")
        #print(f"  dpt_feats: {dpt_feats.shape}")
        #print(f"  mask:      {mask.shape}")
        b, n_views, n_tokens, v_dim = vit_feats.shape
        vit_flat = vit_feats.reshape(-1, n_tokens, v_dim)
        
        # [B*3, 1296, dpt_dim]
        #dpt_flat = dpt_feats.reshape(-1, dpt_feats.shape[-2], dpt_feats.shape[-1])
        b_d, v_d, h, w, d_dim = dpt_feats.shape
        dpt_flat = dpt_feats.reshape(b_d * v_d, h * w, d_dim)
        # 2. 应用 Mask 过滤 (关键步骤)
        # [B, 3] -> [B*3]
        mask_flat = mask.reshape(-1)
        #print(f"  mask_flat shape: {mask_flat.shape}")
        #print(f"  dpt_flat shape:  {dpt_flat.shape}")
        # 只保留 mask 为 True 的样本
        # [N_Valid, 256, vit_dim]
        vit_valid = vit_flat[mask_flat]
        # [N_Valid, 1296, dpt_dim]
        dpt_valid = dpt_flat[mask_flat]
        
        if vit_valid.shape[0] == 0:
            raise ValueError("No valid samples found in the batch for SpatialForcingLoss.")
        
        # 3. 执行投影 (只对有效图片做，省算力！)
        # [N_Valid, 256, vit] -> [N_Valid, 1296, dpt]
        vit_aligned = self.projector(vit_valid)
        
        # 4. 计算 Loss
        # 展平 [N_Valid * 1296, Dim]
        s_vec = vit_aligned.reshape(-1, vit_aligned.shape[-1])
        t_vec = dpt_valid.reshape(-1, dpt_valid.shape[-1])
        
        # 目标是相似度为 1
        target = torch.ones(s_vec.shape[0], device=s_vec.device)
        
        return self.loss_fn(s_vec, t_vec, target)
