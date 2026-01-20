import torch
import numpy as np
from PIL import Image
import comfy.model_management

class FluxKleinTripleConditioning:
    """
    FLUX.2 [klein] 三图条件编码节点
    同时输入三张图片，仅输出CONDITIONING
    使用image1的尺寸作为输出尺寸
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image1": ("IMAGE",),  # 主图像 - 决定输出尺寸
                "image2": ("IMAGE",),  # 参考图像1
                "image3": ("IMAGE",),  # 参考图像2
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "基于参考图像进行编辑",
                }),
                "strength2": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "strength3": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_triple_images"
    CATEGORY = "flux_klein/conditioning"
    
    def __init__(self):
        self.device = comfy.model_management.get_torch_device()
    
    def tensor_to_pil(self, image_tensor):
        """将tensor转换为PIL图像，不进行任何尺寸修改"""
        try:
            # 处理批次维度
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor.squeeze(0)
            
            # 确保在CPU上并转换为numpy
            if image_tensor.device != torch.device('cpu'):
                image_tensor = image_tensor.cpu()
            
            # 转换0-1范围到0-255
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).clamp(0, 255).byte()
            
            # 转换为numpy数组
            if image_tensor.dtype != torch.uint8:
                image_tensor = image_tensor.byte()
            
            # 转换为PIL
            image_array = image_tensor.numpy()
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # RGB图像
                return Image.fromarray(image_array.astype('uint8'), 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
                # 单通道图像
                return Image.fromarray(image_array[:, :, 0].astype('uint8'), 'L')
            else:
                # 默认处理方式
                return Image.fromarray(image_array.astype('uint8'))
        except Exception as e:
            print(f"Error in tensor_to_pil: {e}")
            raise
    
    def encode_image_to_latent(self, vae, image_tensor):
        """将图像编码为latent"""
        try:
            # 转换为PIL
            pil_image = self.tensor_to_pil(image_tensor)
            
            # 获取原始尺寸
            width, height = pil_image.size
            
            # 转换为tensor
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_torch = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
            
            # 调整维度顺序 (H, W, C) -> (B, C, H, W)
            if len(image_torch.shape) == 4:
                if image_torch.shape[-1] == 3:  # (B, H, W, C)
                    image_torch = image_torch.permute(0, 3, 1, 2)
            else:  # (H, W, C) -> (B, C, H, W)
                image_torch = image_torch.permute(2, 0, 1).unsqueeze(0)
            
            # 使用VAE编码
            latent = vae.encode(image_torch)
            
            return latent, (width, height)
        except Exception as e:
            print(f"Error in encode_image_to_latent: {e}")
            raise
    
    def encode_prompt_with_images(self, clip, prompt, latent1, latent2, latent3, 
                                   strength2, strength3):
        """编码提示词并融合图像信息，返回CONDITIONING格式"""
        try:
            # 构建图像感知的提示词
            enhanced_prompt = f"[MAIN IMAGE]: Primary content to edit (size: {list(latent1.shape)})\n"
            enhanced_prompt += f"[REF IMAGE 1]: Reference with strength {strength2}\n"
            enhanced_prompt += f"[REF IMAGE 2]: Reference with strength {strength3}\n"
            
            if prompt and prompt.strip():
                enhanced_prompt += f"\n[INSTRUCTION]: {prompt}"
            
            # 使用CLIP编码
            tokens = clip.tokenize(enhanced_prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            # 构建CONDITIONING格式
            conditioning = [[cond, {"pooled_output": pooled}]]
            
            # 添加多参考图像信息到extra_patches
            conditioning[0][1]["extra_patches"] = {
                "ref_latent_2": {
                    "samples": latent2,
                    "strength": strength2
                },
                "ref_latent_3": {
                    "samples": latent3,
                    "strength": strength3
                }
            }
            
            # 添加主latent的尺寸信息，用于后续解码
            conditioning[0][1]["main_latent_shape"] = list(latent1.shape)
            conditioning[0][1]["reference_count"] = 2  # 参考图像数量
            
            return conditioning
        except Exception as e:
            print(f"Error in encode_prompt_with_images: {e}")
            raise
    
    def encode_triple_images(self, clip, vae, image1, image2, image3, 
                           prompt="", strength2=1.0, strength3=1.0):
        """主函数：编码三张图像并生成CONDITIONING"""
        try:
            # 编码三张图像到latent空间
            latent1_main, size1 = self.encode_image_to_latent(vae, image1)
            latent2_ref, size2 = self.encode_image_to_latent(vae, image2)
            latent3_ref, size3 = self.encode_image_to_latent(vae, image3)
            
            # 编码提示词并生成CONDITIONING
            conditioning = self.encode_prompt_with_images(
                clip, prompt, latent1_main, latent2_ref, latent3_ref, strength2, strength3
            )
            
            # 在CONDITIONING中存储尺寸信息（用于后续节点）
            conditioning[0][1]["size_info"] = {
                "main": size1,
                "ref1": size2,
                "ref2": size3
            }
            
            return (conditioning,)
        except Exception as e:
            print(f"Error in encode_triple_images: {e}")
            raise


# 节点注册
NODE_CLASS_MAPPINGS = {
    "FluxKleinTripleConditioning": FluxKleinTripleConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKleinTripleConditioning": "Flux Klein Triple Conditioning",
}

# 确保模块级别的导入正确
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
