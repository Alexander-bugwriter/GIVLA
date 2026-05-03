import sys
import os
import math
from torch import amp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import wandb
import swanlab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from scripts.GIVLA import GIVLA
from accelerate import Accelerator 
import logging
from datetime import datetime
import argparse
from accelerate import Accelerator, DistributedType
import json
import shutil
from torch.optim import AdamW
from SF_utils import SpatialForcingLoss

import warnings
import contextlib
import torch


accelerator = Accelerator()

def get_with_warning(config: dict, key: str, default):
    if key in config:
        return config[key]
    else:
        warnings.warn(f"'{key}' not found in config, using default: {default!r}")
        return default


def inspect_named_submodules(module_dict: dict, verbose: bool = True):

    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)


def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    state_mask = torch.stack([item["state_mask"] for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)

    return {
        "prompts": prompts,
        "images": images,
        "states": states,
        "actions": actions,
        "action_mask": action_mask,
        "state_mask": state_mask,
        "image_masks": image_masks,
        "embodiment_ids": embodiment_ids
    }

def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step  
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda
    
def setup_logging(log_dir: str) -> str:
    from datetime import datetime
    import logging, os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")
    return log_path

def init_wandb(config: dict, accelerator: Accelerator):

    if accelerator.is_main_process:
        if get_with_warning(config, "disable_wandb", False):
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(
            project=get_with_warning(config, "wandb_project", "default_run"),
            name=get_with_warning(config, "run_name", "default_run"),
            config=config,
            dir=get_with_warning(config, "save_dir", "checkpoints"),
            mode="offline",
        )

        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

def init_swanlab(config: dict, accelerator: Accelerator):

    if accelerator is None or accelerator.is_main_process:
        swanlab.init(
            project=config.get("wandb_project", "default_run"),
            name=config.get("run_name", "default_run"),
            config=config
        )

def prepare_dataset(config: dict) -> torch.utils.data.Dataset:
    dataset_type = get_with_warning(config, "dataset_type", "lerobot")
    image_size = get_with_warning(config, "image_size", 448)
    max_samples = get_with_warning(config, "max_samples_per_file", None)
    horizon = get_with_warning(config, "horizon", 50)
    binarize_gripper = get_with_warning(config, "binarize_gripper", False)
    use_augmentation = get_with_warning(config, "use_augmentation", False)
    if dataset_type == "lerobot":
        from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset 
        import yaml
        with open(config.get("dataset_config_path"), 'r') as f:
            dataset_config = yaml.safe_load(f)

        dataset = LeRobotDataset(
            config=dataset_config,
            image_size=image_size,
            max_samples_per_file=max_samples,
            action_horizon=horizon,
            binarize_gripper=binarize_gripper,
            use_augmentation=use_augmentation
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Loaded {len(dataset)} samples from {config['data_paths']} ({dataset_type})")
    return dataset


def prepare_dataloader(dataset, config: dict) -> DataLoader:
    batch_size = get_with_warning(config, "batch_size", 8)
    num_workers = get_with_warning(config, "num_workers", 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Initialized dataloader with batch size {batch_size}")
    return dataloader


def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info(f"[Step {step}] Non-finite detected in {name}")
            return False
    return True

def log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator,task_loss=None, align_loss=None):
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        logging.info(f"[Step {step}] Loss: {loss.item():.4f}")
        log_dict = {
            "step": step,
            "total_loss": loss.item(), # 建议改名叫 total_loss 更清晰
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        # 【新增】如果有 task_loss 就记录
        if task_loss is not None:
            log_dict["task_loss"] = task_loss.item()
            
        # 【新增】如果有 align_loss 就记录
        if align_loss is not None:
            log_dict["align_loss"] = align_loss.item()
        wandb.log(log_dict)
        swanlab.log(log_dict)

def save_checkpoint(save_dir, step, model_engine, loss, accelerator, config=None, norm_stats=None ):
    tag = f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, tag)

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    client_state = {
        "step": step,
        "best_loss": loss if isinstance(loss, float) else loss.item(),
        "config": config,
    } if accelerator.is_main_process else {} 

    model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
    if accelerator.is_main_process:
        if config is not None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        if norm_stats is not None:
            norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.json")
            with open(norm_stats_path, "w") as f:
                json.dump(norm_stats, f, indent=2)
                
        checkpoint_meta_path = os.path.join(checkpoint_dir, "checkpoint.json")
        checkpoint_meta = {
            "type": "ds_model",
            "version": 0.0,
            "checkpoints": "mp_rank_00_model_states.pt"
        }
        with open(checkpoint_meta_path, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)
        logging.info(f"[Rank {accelerator.process_index}] Saved checkpoint to {checkpoint_dir}")

def load_checkpoint_with_deepspeed(model_engine, load_dir, accelerator, tag="step_best", load_optimizer_states=True, resume_pretrain=False):

    try:
        load_path, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=load_optimizer_states and not resume_pretrain,
            load_lr_scheduler_states=load_optimizer_states and not resume_pretrain
        )
        if accelerator.is_main_process:
            logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (including optimizer states)")
        return client_state.get("step", 0), client_state
        
    except Exception as e:
        if accelerator.is_main_process:
            logging.warning(f"World size mismatch detected: {str(e)}")
            logging.warning("Attempting to load only model weights (skipping optimizer states)...")
        try:
            load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False
            )
            if accelerator.is_main_process:
                logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (model weights only)")
            return client_state.get("step", 0), client_state
            
        except Exception as e2:
            if accelerator.is_main_process:
                logging.error(f"Failed to load checkpoint even without optimizer states: {str(e2)}")
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir} with tag {tag}: {str(e2)}")

    

def get_and_clip_grad_norm(accelerator, model, loss, max_norm: float = 1.0):

    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
       
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
 
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(grad_norms) == 0:
            total_norm = torch.tensor(0.0, device=loss.device)
        else:
            total_norm = torch.norm(torch.stack(grad_norms), 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        clipped_grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(clipped_grad_norms) == 0:
            clipped_norm = torch.tensor(0.0, device=loss.device)
        else:
            clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2)

    return total_norm, clipped_norm

def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        (no_decay if is_bias or is_norm else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]



@contextlib.contextmanager
def suspend_deepspeed_allreduce(model):
    """
    临时禁用 DeepSpeed ZeRO optimizer 上的 _grad_acc_post_hooks，
    防止 torch.autograd.grad() 触发 allreduce，用完自动恢复。

    用法：
        with suspend_deepspeed_allreduce(model):
            grads = torch.autograd.grad(loss, params, ...)

    兼容性说明：
        _grad_acc_post_hooks 是 DeepSpeed base_optimizer.py 中的公开属性，
        自 DeepSpeed 0.9.x 起稳定存在。若未来版本改名，fallback 为 nullcontext。
    """
    # DeepSpeed engine 直接挂在 accelerate 包装的 model 上
    # 对应关系：accelerate model → DS engine；DS engine.optimizer → ZeRO optimizer
    ds_optimizer = getattr(model, 'optimizer', None)

    if ds_optimizer is None or not hasattr(ds_optimizer, '_grad_acc_post_hooks'):
        # 找不到 hook 列表时安全退出，不报错
        yield
        return

    # 摘除 hooks（备份 → 清空 → 恢复）
    saved_hooks = list(ds_optimizer._grad_acc_post_hooks)
    ds_optimizer._grad_acc_post_hooks.clear()
    try:
        yield
    finally:
        ds_optimizer._grad_acc_post_hooks.extend(saved_hooks)

def analyze_gradient_conflict_manual_recompute_gemini_2(model, action_loss, align_loss, accelerator):
    """
    使用 torch.autograd.grad 分析梯度冲突。
    None 梯度将被处理为全 0 张量，以保持向量维度严格对齐。
    """
    # 1. 筛选需要分析的参数 (FiLM 层)
    target_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("film" in name.lower() or "dpt_linear" in name.lower()):
            target_params.append(param)

    if not target_params:
        return

    # 2. 分别计算两个 Loss 的梯度
    # retain_graph=True 保证在同一次 forward 后可以多次计算梯度
    # allow_unused=True 允许某些参数不出现在某个 loss 的计算图中
    with suspend_deepspeed_allreduce(model):
        grads_action = torch.autograd.grad(
            outputs=action_loss,
            inputs=target_params,
            retain_graph=True,
            allow_unused=True,
            # 加上这个可以强制不触发某些 hook
            only_inputs=True,
        )

        grads_align = torch.autograd.grad(
            outputs=align_loss,
            inputs=target_params,
            retain_graph=True,
            allow_unused=True,
            # 加上这个可以强制不触发某些 hook
            only_inputs=True,
        )

    # 3. 处理 None 并展平梯度
    flat_action_list = []
    flat_align_list = []

    for p, g_act, g_ali in zip(target_params, grads_action, grads_align):
        # 如果 Action Loss 对该参数没贡献，创建同形状的全 0 Tensor
        if g_act is None:
            flat_action_list.append(torch.zeros_like(p).flatten())
        else:
            flat_action_list.append(g_act.detach().flatten())

        # 如果 Align Loss 对该参数没贡献，创建同形状的全 0 Tensor
        if g_ali is None:
            flat_align_list.append(torch.zeros_like(p).flatten())
        else:
            flat_align_list.append(g_ali.detach().flatten())

    # 4. 拼接成最终的分析向量
    vec_action = torch.cat(flat_action_list)
    vec_align = torch.cat(flat_align_list)

    # 5. 计算指标
    dot_product = torch.dot(vec_action, vec_align)
    norm_action = vec_action.norm()
    norm_align = vec_align.norm()
    
    # 计算余弦相似度
    # cos = (A · B) / (|A| * |B|)
    denominator = norm_action * norm_align
    if denominator > 1e-12:
        cosine_sim = dot_product / denominator
    else:
        cosine_sim = torch.tensor(0.0).to(vec_action.device)

    # 6. 记录日志 (仅在主进程)
    if accelerator.is_main_process:
        # 控制台打印，方便实时观察
        print(f"\n>>> [Grad Conflict Analysis] <<<")
        print(f"    Cos-Sim: {cosine_sim.item():.6f} | Dot: {dot_product.item():.6e}")
        print(f"    Norm (Action/Align): {norm_action.item():.4f} / {norm_align.item():.4f}")

        swanlab.log({
            "grad_analysis/cosine_similarity": cosine_sim.item(),
            "grad_analysis/action_norm": norm_action.item(),
            "grad_analysis/align_norm": norm_align.item(),
            "grad_analysis/dot_product": dot_product.item()
        })

    # 函数结束，不涉及 backward()，不会干扰 DeepSpeed 的规约

def train(config):


    # === Set logging ===
    save_dir = get_with_warning(config, "save_dir", "checkpoints")
    log_path = setup_logging(save_dir)
    
    # === WandB and Swanlab ===
    init_wandb(config, accelerator)
    init_swanlab(config, accelerator)

    # === Debug mode ===
    if get_with_warning(config, "debug", False):
        torch.autograd.set_detect_anomaly(True)

    # === Dataset ===
    dataset = prepare_dataset(config)

    # === DataLoader ===
    dataloader = prepare_dataloader(dataset, config)

    # === Model ===
    model = GIVLA(config)
    model.train()
    model.set_finetune_flags()
    # ================= [修改] Spatial Forcing 初始化 =================
    use_align = get_with_warning(config, "SF_align",False)
    print(f"Spatial Forcing Align Loss Enabled: {use_align}")
    sf_loss_module = None
    
    # [逻辑修改] 只要开启了 use_align，就初始化 SF 模块
    # 不再强制要求 model.da3_film_in 为 True
    if use_align:
        logging.info("Initializing Spatial Forcing Module (Decoupled Mode)...")
        if not model.da3_film_in:
            logging.info("-> Mode: SF_OUT Only (Align Loss ON, FiLM OFF)")
        else:
            logging.info("-> Mode: Full SF (Align Loss ON, FiLM ON)")
        sf_loss_module = model.sf_loss_module

    lr = get_with_warning(config, "lr", 1e-5)
    wd = get_with_warning(config, "weight_decay", 1e-5)
    optimizer = AdamW(build_param_groups(model, wd), lr=lr) # 使用包含 SF 参数的 params
    if accelerator.is_main_process:
        logging.info(f"Optimizer=AdamW, lr={lr}, weight_decay={wd}")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model  
  
    if accelerator.is_main_process:
        logging.info("Initialized with Accelerate")
    
    
    # === Warmup + Cosine Scheduler ===
    max_steps = get_with_warning(config, "max_steps", 1000)
    warmup_steps = get_with_warning(config, "warmup_steps", 300)
    
    # === loss function ===
    loss_fn = nn.MSELoss() 

    # === Checkpoint and save path setup ===
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pt")
    best_loss = float("inf")
    
    # === Logging and interval settings ===
    log_interval = get_with_warning(config, "log_interval", 100)
    ckpt_interval = get_with_warning(config, "ckpt_interval", 1000)
    max_norm = get_with_warning(config, "grad_clip_norm", 1.0)

    # === Resume training from checkpoint ===
    resume = get_with_warning(config, "resume", False)
    resume_path = get_with_warning(config, "resume_path", None)
    resume_pretrain = get_with_warning(config, "resume_pretrain", False)

    if resume != bool(resume_path):
        raise ValueError("Inconsistent resume configuration: --resume and --resume_path must be set together.")
    
    if resume:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)

        step, client_state = load_checkpoint_with_deepspeed(
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            load_optimizer_states=True,  
            resume_pretrain=resume_pretrain
        )
        best_loss = client_state.get("best_loss", float("inf"))
        if accelerator.is_main_process:
            logging.info(f"Resuming from {resume_dir}/{resume_tag}, step {step}")
    else:
        step = 0
        if accelerator.is_main_process:
            logging.info("Starting fresh training")

    if resume_pretrain:
        step = 0
        logging.info("Resuming pretraining from scratch, resetting step to 0")

    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))


    if accelerator.is_main_process:
        
        inspect_named_submodules({
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_head": model.action_head
        })

    # === Training Loop ===
    while step < max_steps:
        for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break
            prompts = batch["prompts"]
            images_batch = batch["images"]
            image_masks = batch["image_masks"]
            states = batch["states"].to(dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(dtype=torch.bfloat16)
            action_mask = batch["action_mask"]
            state_mask = batch["state_mask"]
            embodiment_ids = batch["embodiment_ids"]
            fused_tokens_list = []
            align_feats_list = [] # 【新增】用于收集对齐特征
            align_masks_list = [] # 【新增】收集 Mask
            
            for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
                result = model.get_vl_embeddings(images=images, image_mask=image_mask, prompt=prompt, return_cls_only=False,return_align_feats=use_align )
                # 【新增】处理返回结果 (Tuple or Tensor)
                if use_align and isinstance(result, tuple):
                    fused, align_feats = result
                    align_feats_list.append(align_feats)
                    align_masks_list.append(image_mask) # 【新增】保存对应的 mask [3]
                else:
                    fused = result
                    
                fused_tokens_list.append(fused.to(dtype=torch.bfloat16))
            
            fused_tokens = torch.cat(fused_tokens_list, dim=0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                pred_velocity, noise = model(fused_tokens, state=states, actions_gt=actions_gt, action_mask=action_mask)
                
            target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
            
            assert pred_velocity.shape == target_velocity.shape

            if action_mask.sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
                            f"This indicates a problem with the data or mask generation. "
                            f"action_mask shape: {action_mask.shape}, "
                            f"action_mask: {action_mask}")
            
            action_mask_original = action_mask  # 保存原始形状 [16, 50, 24]
            action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
            pred_velocity_mask = pred_velocity * action_mask
            task_loss = loss_fn(pred_velocity_mask, target_velocity)
            scale_factor = action_mask.numel() / (action_mask.sum() + 1e-8)
            task_loss = task_loss * scale_factor
            
            # ================= [新增] 计算 Align Loss =================
            align_loss_val = torch.tensor(0.0, device=accelerator.device)
            
            
            if use_align and sf_loss_module is not None and len(align_feats_list) > 0:
                # 拼接 Batch 中的特征
                vlm_feats = torch.cat([f['vlm_feat'] for f in align_feats_list if 'vlm_feat' in f], dim=0)
                da3_feats = torch.cat([f['da3_feat'] for f in align_feats_list if 'da3_feat' in f], dim=0)
                mask = torch.stack(align_masks_list, dim=0).to(accelerator.device)
                
                # 计算 SF Loss
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    align_loss_val = sf_loss_module(vlm_feats, da3_feats, mask=mask)
                
                # 加权求和
                alpha = get_with_warning(config, "align_loss_alpha", 0.5)
                total_loss = task_loss + alpha * align_loss_val
            else:
                total_loss = task_loss

            
            # === NaN/Inf check ===
            if not check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                pred_velocity=pred_velocity,
                loss=total_loss
            ):
                continue
            # 在这里插入日志逻辑
            #if use_align and step % log_interval == 0:
            #    analyze_gradient_conflict_manual_recompute_gemini(
            #        model=model,
            #        prompts=prompts,
            #        images_batch=images_batch,
            #        image_masks=image_masks,
            #        states=states,
            #        actions_gt=actions_gt,
            #        action_mask_original=action_mask_original,
            #        loss_fn=loss_fn,
            #        sf_loss_module=sf_loss_module,
            #        optimizer=optimizer,
            #        accelerator=accelerator,
            #        step=step,
            #        use_align=use_align
            #    )
            if step % log_interval == 0:
                with accelerator.no_sync(model):
                    analyze_gradient_conflict_manual_recompute_gemini_2(
                        model=model,
                        action_loss=task_loss,      # 传入拆分后的 task_loss
                        align_loss=alpha * align_loss_val,  # 传入拆分后的 align_loss
                        accelerator=accelerator
                    )


            # === Backward and optimizer step ===
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(total_loss)

            # === Clip grad norm ===
            total_norm, clipped_norm = get_and_clip_grad_norm(accelerator, model, total_loss, max_norm)

            optimizer.step()
            scheduler.step()
            
            # === Logging ===
            if step % log_interval == 0:
                log_training_step(step, total_loss, total_norm, clipped_norm, scheduler, dataloader, accelerator,task_loss=task_loss,  align_loss=align_loss_val if use_align else None)
   
            # === Save best checkpoint ===
            loss_value = total_loss.item()
            if accelerator.is_main_process:
                is_best = loss_value < best_loss
                if is_best:
                    best_loss = loss_value
                is_best_tensor = torch.tensor(int(is_best), device=accelerator.device)
            else:
                is_best_tensor = torch.tensor(0, device=accelerator.device)
            
            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)
            
            if is_best_tensor.item() == 1 and step > 1000:
                accelerator.print("start to save best checkpoint")
                save_checkpoint(
                    save_dir,
                    step="best",
                    model_engine=model_engine,
                    loss=total_loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=dataset.arm2stats_dict,
                )
                accelerator.print("end to save best checkpoint")
                if accelerator.is_main_process:
                    logging.info(f"Saved best checkpoint at step {step} with loss {loss_value:.6f}")

            step += 1

            # === Save periodic checkpoint ===
            if step % ckpt_interval == 0 and step > 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
                save_checkpoint(save_dir, step=step, model_engine=model_engine, loss=total_loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict, )
         
    # === Save final model ===
    save_checkpoint(save_dir, step="final", model_engine=model_engine, loss=total_loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict, )
    logging.info(f"Final model saved to step_final/")
    logging.info(f"Best checkpoint saved to step_best/ with loss {best_loss:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Evo-1")

    # Basic config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching"])
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="lerobot")
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--binarize_gripper", action="store_true", default=False, help="Whether to binarize gripper state/action (default: False).")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation on images")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--da3_film_in", action="store_true")
    parser.add_argument("--SF_align", action="store_true", help="Enable Spatial Forcing alignment loss")
    parser.add_argument("--align_loss_alpha", type=float, default=0.5, help="Weight for alignment loss")


    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # Resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")
   

    # Finetuning
    parser.add_argument("--finetune_vit", action="store_true")
    parser.add_argument("--finetune_llm_backbone", action="store_true")
    parser.add_argument("--finetune_dpt_linear_film", action="store_true")
    parser.add_argument("--finetune_action_head", action="store_true")
    parser.add_argument("--finetune_da3", action="store_true")
    parser.add_argument("--finetune_SF_align", action="store_true")

    # Misc
    parser.add_argument("--per_action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    # dropout
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    config = vars(args)

    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)

