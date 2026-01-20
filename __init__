# flux_klein_dual_edit.py
import torch
import numpy as np
from PIL import Image
import folder_paths
from comfy.model_management import get_torch_device
from comfy.sd import load_checkpoint_guess_config
from comfy.utils import common_upscale
import comfy.samplers as samplers

class FluxKleinDualImageEdit:
    """
    FLUX.2 [klein] 双图编辑节点
    同时输入两张图片，使图片1参考图片2的内容生成
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image1": ("IMAGE",),  # 主图像 - 将被编辑
                "image2": ("IMAGE",),  # 参考图像 - 提供参考内容
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "根据参考图像的风格/内容编辑第一张图像"
                }),
                "target_size": ("INT", {
                    "default": 896,
                    "min": 256,
                    "max": 2048,
                    "step": 32,
                    "tooltip": "输出图像的目标尺寸，建议为输出最长边的0.9倍"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "参考图像对生成结果的影响强度"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "crop_to_fit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否裁剪图像以适应目标尺寸"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("latent", "ref_latent", "encoded_prompt")
    FUNCTION = "encode_dual_images"
    CATEGORY = "flux_klein/edit"
    
    def __init__(self):
        self.device = get_torch_device()
    
    def prepare_images(self, image1, image2, target_size, crop_to_fit):
        """准备和预处理图像"""
        
        # 转换tensor为PIL图像
        if len(image1.shape) == 4:
            image1 = image1.squeeze(0)
        if len(image2.shape) == 4:
            image2 = image2.squeeze(0)
        
        # 将0-1范围转换为0-255
        if image1.max() <= 1.0:
            image1 = (image1 * 255).clamp(0, 255).byte()
        if image2.max() <= 1.0:
            image2 = (image2 * 255).clamp(0, 255).byte()
        
        # 转换为PIL
        i1 = Image.fromarray(image1.cpu().numpy())
        i2 = Image.fromarray(image2.cpu().numpy())
        
        # 确保两张图像尺寸一致
        if i1.size != i2.size:
            # 统一调整为较小图像的尺寸或目标尺寸
            if crop_to_fit:
                # 计算裁剪区域
                min_width = min(i1.width, i2.width, target_size)
                min_height = min(i1.height, i2.height, target_size)
                
                # 居中裁剪
                left1 = (i1.width - min_width) // 2
                top1 = (i1.height - min_height) // 2
                left2 = (i2.width - min_width) // 2
                top2 = (i2.height - min_height) // 2
                
                i1 = i1.crop((left1, top1, left1 + min_width, top1 + min_height))
                i2 = i2.crop((left2, top2, left2 + min_width, top2 + min_height))
            else:
                # 缩放至目标尺寸
                size = (target_size, target_size)
                i1 = i1.resize(size, Image.LANCZOS)
                i2 = i2.resize(size, Image.LANCZOS)
        
        return i1, i2
    
    def encode_images(self, vae, image_pil):
        """使用VAE编码图像到latent空间"""
        
        # 转换为tensor
        image_tensor = torch.from_numpy(np.array(image_pil).astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 调整维度顺序 (B, H, W, C) -> (B, C, H, W)
        if len(image_tensor.shape) == 4 and image_tensor.shape[-1] == 3:
            image_tensor = image_tensor.permute(0, 3, 1, 2)
        
        # 使用VAE编码
        latent = vae.encode(image_tensor)
        
        return {"samples": latent}
    
    def encode_prompt(self, clip, prompt, image1, image2):
        """编码文本提示和图像"""
        
        # FLUX klein使用特殊的提示编码方式
        # 参考qwen edit的设计，将图像信息融入提示
        
        full_prompt = f"[IMAGE1]: Original image to edit\n"
        full_prompt += f"[IMAGE2]: Reference image for style/content\n"
        full_prompt += f"[PROMPT]: {prompt}"
        
        # 使用CLIP编码
        tokens = clip.tokenize(full_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        return cond, pooled
    
    def encode_dual_images(self, model, clip, vae, image1, image2, prompt, 
                          target_size, strength, negative_prompt="", crop_to_fit=True):
        """主函数：编码双图像并准备latent"""
        
        # 准备图像
        img1_pil, img2_pil = self.prepare_images(image1, image2, target_size, crop_to_fit)
        
        # 编码主图像到latent
        latent_main = self.encode_images(vae, img1_pil)
        
        # 编码参考图像到latent
        latent_ref = self.encode_images(vae, img2_pil)
        
        # 编码文本提示
        cond, pooled = self.encode_prompt(clip, prompt, img1_pil, img2_pil)
        
        # 在latent中存储额外信息
        # 参考Flux的工作方式，在extra_patches中添加参考信息
        latent_main["extra_patches"] = {
            "ref_latent": latent_ref["samples"],
            "strength": strength,
            "prompt_embeds": cond,
            "pooled_prompt_embeds": pooled,
            "negative_prompt_embeds": clip.tokenize(negative_prompt) if negative_prompt else None
        }
        
        # 返回latent和参考latent
        return (latent_main, latent_ref, prompt)


class FluxKleinDualEditSampler:
    """
    专为FluxKleinDualImageEdit设计的采样器
    处理双图像编辑的特殊逻辑
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 20, "tooltip": "Flux Klein使用4步采样效果最佳"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpmpp_2m", "dpmpp_3m_sde"],),
                "scheduler": (["normal", "karras", "exponential", "simple", "ddim_uniform"],),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "flux_klein/sampling"
    
    def sample(self, model, latent, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise):
        """自定义采样过程"""
        
        # 从extra_patches获取参考信息
        ref_latent = None
        strength = 1.0
        
        if "extra_patches" in latent:
            ref_latent = latent["extra_patches"].get("ref_latent")
            strength = latent["extra_patches"].get("strength", 1.0)
        
        # 使用标准的ComfyUI采样器
        # 但注入参考latent信息
        if ref_latent is not None:
            # 在采样过程中混合参考latent
            # 这里使用简单的加权平均，实际可以更复杂
            main_samples = latent["samples"]
            ref_samples = ref_latent
            
            # 确保尺寸匹配
            if main_samples.shape != ref_samples.shape:
                ref_samples = torch.nn.functional.interpolate(
                    ref_samples, size=main_samples.shape[2:], mode='bilinear'
                )
            
            # 混合latent
            mixed_samples = (1 - strength) * main_samples + strength * ref_samples
            latent["samples"] = mixed_samples
        
        # 使用标准采样器
        samples = common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler, 
            positive, negative, latent, denoise=denoise
        )
        
        return (samples, )


# 辅助函数：标准KSampler
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                   positive, negative, latent, denoise=1.0, disable_noise=False):
    """ComfyUI标准采样器"""
    
    from comfy.sample import prepare_noise, prepare_sampling
    from comfy.samplers import KSampler as RealKSampler
    
    device = get_torch_device()
    
    # 准备latent
    samples = latent["samples"]
    
    if not disable_noise:
        noise = prepare_noise(samples, seed, None)
    else:
        noise = torch.zeros_like(samples)
    
    # 准备采样器
    sampler = RealKSampler(model, steps, device)
    
    # 采样
    samples = sampler.sample(
        noise, 
        positive, 
        negative, 
        cfg=cfg,
        latent_image=samples,
        start_step=0,
        last_step=steps,
        force_full_denoise=True,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=seed
    )
    
    out = latent.copy()
    out["samples"] = samples
    
    return out


# 节点注册
NODE_CLASS_MAPPINGS = {
    "FluxKleinDualImageEdit": FluxKleinDualImageEdit,
    "FluxKleinDualEditSampler": FluxKleinDualEditSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKleinDualImageEdit": "Flux Klein Dual Image Edit",
    "FluxKleinDualEditSampler": "Flux Klein Dual Edit Sampler",
}
