# model/internvl3/internvl3_embedder.py
import torch
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig
from torchvision.transforms.functional import to_pil_image
from typing import Union, List
from torch import nn
import logging
from safetensors.torch import load_file
from model.depth_anything_v3.src.depth_anything_3.api import DepthAnything3
from torch.utils.checkpoint import checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# === Aspect Ratio Handling ===
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVL3Embedder(nn.Module):
    def __init__(self, model_name="OpenGVLab/InternVL3-1B", image_size=448, device="cuda",da3_film_in=False):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.transform = build_transform(image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.da3_film_in = da3_film_in
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = False
          
        self.dpt_embedder = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
        self.dpt_embedder.to(self.device)
        class DPTTokenwiseFilM(nn.Module):
            """Token-wise FiLM - 批量处理版本"""
            def __init__(self, dim_in=1536, dim_out=896, vit_num_tokens=256, dpt_num_tokens=1297,init_gamma_scale=0.01, init_beta_scale=0.01):
                super().__init__()
                self.vit_num_tokens = vit_num_tokens
                self.dpt_num_tokens = dpt_num_tokens
                # 降维
                self.proj = nn.Linear(dim_in, dim_out)
                self.proj_norm = nn.LayerNorm(dim_out)
                self.proj_act = nn.ReLU(inplace=True)
                # Token压缩
                self.token_mapper = nn.Linear(dpt_num_tokens, vit_num_tokens)
                self.token_norm = nn.LayerNorm(dim_out)
                self.token_act = nn.ReLU(inplace=True)
                # FiLM生成
                # self.proj_c = nn.Linear(dim_out, dim_out * 2)
                # ✅ FiLM生成器：分成两个独立的Linear
                self.gamma_generator = nn.Linear(dim_out, dim_out)
                self.beta_generator = nn.Linear(dim_out, dim_out)
                # ✅ 初始化为接近0的值
                self._init_film_generators(init_gamma_scale, init_beta_scale)
            def _init_film_generators(self, gamma_scale=0.01, beta_scale=0.01):
                """
                初始化FiLM生成器
                - gamma_generator: 小初始化 → gamma ≈ 1
                - beta_generator: 更小初始化 → beta ≈ 0
                """
                # Gamma生成器：权重和偏置都接近0
                nn.init.normal_(self.gamma_generator.weight, mean=0.0, std=gamma_scale)
                nn.init.zeros_(self.gamma_generator.bias)
                # Beta生成器：权重和偏置都更接近0
                nn.init.normal_(self.beta_generator.weight, mean=0.0, std=beta_scale)
                nn.init.zeros_(self.beta_generator.bias)
            def forward(self, dpt_embeds):
                """
                Args:
                    dpt_embeds: [1, 3, 1297, 1536] 或 [3, 1297, 1536]
                
                Returns:
                    gamma: [3, 256, 896]
                    beta:  [3, 256, 896]
                """
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 确保是 [3, 1297, 1536]
                    if dpt_embeds.dim() == 4 and dpt_embeds.shape[0] == 1:
                        dpt_embeds = dpt_embeds.squeeze(0)
                    # 降维
                    x = self.proj(dpt_embeds)       # [3, 1297, 1536] → [3, 1297, 896]
                    x = self.proj_act(x)
                    x = self.proj_norm(x)
                    # Token压缩
                    x = x.transpose(1, 2)           # [3, 1297, 896] → [3, 896, 1297]
                    x = self.token_mapper(x)        # [3, 896, 1297] → [3, 896, 256]
                    x = x.transpose(1, 2)           # [3, 896, 256] → [3, 256, 896]
                    x = self.token_act(x)
                    x = self.token_norm(x)
                    # ✅ 分别生成gamma和beta
                    gamma_raw = self.gamma_generator(x)  # [3, 256, 896]
                    beta_raw = self.beta_generator(x)    # [3, 256, 896]
                    # ✅ gamma以1为中心，beta以0为中心
                    gamma = 1.0 + gamma_raw  # gamma ≈ 1 + 小扰动
                    beta = beta_raw           # beta ≈ 0 + 小扰动
                    return gamma, beta
        self.dpt_linear_film = DPTTokenwiseFilM().to(self.device)
    
    def apply_film(self, features, gama, beta):
        
        return features * gama + beta       
    def _preprocess_images(
        self,
        image_tensors: List[Union[Image.Image, torch.Tensor]]
    ) -> (torch.Tensor, List[int]):

        pixel_values_list = []
        for i, image in enumerate(image_tensors):
            if isinstance(image, torch.Tensor):
                image = to_pil_image(image)
            tiles = dynamic_preprocess(image, image_size=self.image_size)
            tile_tensors = torch.stack([self.transform(t) for t in tiles])  # (T_i, 3, 448, 448)
            pixel_values_list.append(tile_tensors)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=torch.bfloat16, device=self.device)
        num_tiles_list = [pv.shape[0] for pv in pixel_values_list]

        return pixel_values, num_tiles_list

    def _build_multimodal_prompt(
        self,
        num_tiles_list: List[int],
        text_prompt: str
    ) -> str:

        prompt = ''
        for i in range(len(num_tiles_list)):
            prompt += f"Image-{i+1}: <image>\n"
        prompt += text_prompt.strip()

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        for tile_count in num_tiles_list:
            token_count = self.model.num_image_token * tile_count
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
            prompt = prompt.replace("<image>", image_tokens, 1)

        return prompt
    
    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int]
    ) -> (torch.Tensor, torch.Tensor):
   
        untruncated_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        true_sequence_length = untruncated_ids.shape[1]

        if true_sequence_length > self.max_text_length:
            print("\n" + "="*80)
            print(f" WARNING: Input prompt was TRUNCATED!")
            print(f"   - Max Length Allowed    : {self.max_text_length}")
            print(f"   - Actual Length      : {true_sequence_length}")
            print(f"   - Truncated Prompt (first 100 chars): '{prompt[:100]}...'")
            print("="*80 + "\n")

        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_text_length).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

       
        img_token_mask = (input_ids == self.img_context_token_id)
     
        img_token_locations = torch.where(img_token_mask)[1]


        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        selected = (input_ids == self.img_context_token_id)

            
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

 
        tokens_per_tile = self.model.num_image_token 
 
        torch.set_printoptions(profile="full", threshold=float('inf'))
   
        torch.set_printoptions(profile="default")
        current_token_idx = 0
        for i in range(len(image_mask)):
           
            num_tiles_for_this_image = num_tiles_list[i]
            num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
       
            if not image_mask[i]:
                
                start_idx = img_token_locations[current_token_idx]
                end_idx = start_idx + num_tokens_for_this_image
               
                attention_mask[0, start_idx:end_idx] = 0
    
            current_token_idx += num_tokens_for_this_image

        input_embeds = input_embeds.reshape(B, N, C)
    
        torch.set_printoptions(profile="full", threshold=float('inf'))
     
        torch.set_printoptions(profile="default")
        return input_embeds, attention_mask, img_token_mask
    
    def _da3_forward(self, images):
        """
        单独写一个 forward 供 checkpoint 调用。
        images 必须是 tuple/list，checkpoint 要求 *args 展开
        """
        # 把 PIL/tensor 统一转成 RGB 的 tensor，DA3 内部需要
        return self.dpt_embedder._get_dpt_embeddings(images)

    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors: list[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        text_prompt: str,
        return_cls_only: bool = True,
        align_layer_idx: int = 7,        # 新增参数
        return_align_feats: bool = False # 新增参数
    ):

   
        pixel_values, num_tiles_list = self._preprocess_images(image_tensors)

       
        if pixel_values.shape[0] == 0:
           
            print("Warning: No valid images to process after masking.")

        vit_embeds = self.model.extract_feature(pixel_values)
        dpt_embeds = checkpoint(
                self._da3_forward,
                image_tensors          # 注意传 tuple/list
            )
        if self.da3_film_in:
            # 生成FiLM参数
            gamma, beta = self.dpt_linear_film(dpt_embeds)  # [3, 1, 896]
            # 应用FiLM调制VIT特征
            fused_embeds = self.apply_film(vit_embeds, gamma, beta)  # [3, 256, 896]
        else:
            fused_embeds = vit_embeds
        # if self.use_da3:
        #     fused_embeds = vit_embeds
        #     dpt_embeds = self.dpt_embedder._get_dpt_embeddings(image_tensors)
        # else:
        #     fused_embeds = vit_embeds  
        prompt = self._build_multimodal_prompt(num_tiles_list, text_prompt)
        inputs_embeds, attention_mask, img_token_mask  = self._prepare_and_fuse_embeddings(prompt, fused_embeds, image_mask, num_tiles_list)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)
        fused_hidden = fused_hidden[:, 0, :] if return_cls_only else fused_hidden

        align_feats = {}
        if return_align_feats :
            #print("inputs_embeds.shape:", inputs_embeds.shape)
            batch_size = inputs_embeds.shape[0]
            # --- A. 提取 VLM 视觉 Token ---
            # 直接切片提取，展平所有 batch 和 seq
            # [Total_Tokens, Dim] -> reshape -> [Batch, 3, 256, Dim]
            align_feats['vlm_feat'] = outputs.hidden_states[align_layer_idx][img_token_mask].view(
                batch_size, 3, 256, -1
            )
            #print("align_feats['vlm_feat'].shape:", align_feats['vlm_feat'].shape)
            # --- B. 提取 DA3 视觉特征 ---
            # 1. 去掉 Global Token (:-1)
            # 2. Reshape 成空间图 (1296 -> 36x36)
            # Input: [Batch, 3, 1297, 1536] -> Output: [Batch, 3, 36, 36, 1536]
            align_feats['da3_feat'] = dpt_embeds[:, :, :-1, :].view(
                batch_size, 3, 36, 36, -1
            )
            #print("align_feats['da3_feat'].shape:", align_feats['da3_feat'].shape)

        if return_align_feats:
            return fused_hidden, align_feats
        return fused_hidden
