# flux_klein_triple_conditioning.py
import torch
import numpy as np
from PIL import Image
import sys
import os

# 确保能导入comfy相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import comfy.model_management
    import folder_paths
    from comfy.model_management import get_torch_device
except ImportError as e:
    print(f"[Flux Klein Triple] Import error: {e}")
    print("[Flux Klein Triple] Please make sure this file is in ComfyUI/custom_nodes/flux_klein_triple/")

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
        try:
            self.device = get_torch_device()
        except:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def tensor_to_pil(self, image_tensor):
        """将tensor转换为PIL图像"""
        try:
            if image_tensor is None:
                raise ValueError("Input image tensor is None")
                
            # 处理批次维度
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor.squeeze(0)
            
            # 确保在CPU上
            if image_tensor.device != torch.device('cpu'):
                image_tensor = image_tensor.cpu()
            
            # 转换0-1范围到0-255
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).clamp(0, 255)
            
            # 转换为byte类型
            image_tensor = image_tensor.byte()
            
            # 转换为numpy数组
            image_array = image_tensor.numpy()
            
            # 转换为PIL
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                return Image.fromarray(image_array, 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                return Image.fromarray(image_array, 'RGBA')
            elif len(image_array.shape) == 2:
                return Image.fromarray(image_array, 'L')
            else:
                return Image.fromarray(image_array)
                
        except Exception as e:
            print(f"[Flux Klein Triple] Error converting tensor to PIL: {e}")
            print(f"Tensor shape: {image_tensor.shape if image_tensor is not None else 'None'}")
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
            
            # 调整维度顺序
            if len(image_torch.shape) == 4 and image_torch.shape[-1] == 3:
                image_torch = image_torch.permute(0, 3, 1, 2)
            
            # 使用VAE编码
            latent = vae.encode(image_torch)
            
            return latent, (width, height)
            
        except Exception as e:
            print(f"[Flux Klein Triple] Error encoding image to latent: {e}")
            raise
    
    def encode_prompt_with_images(self, clip, prompt, latent1, latent2, latent3, 
                                   strength2, strength3):
        """编码提示词并融合图像信息"""
        try:
            # 构建增强提示词
            enhanced_prompt = f"[MAIN]: Edit this image (latent shape: {latent1.shape})\n"
            enhanced_prompt += f"[REF1]: Apply with strength {strength2}\n"
            enhanced_prompt += f"[REF2]: Apply with strength {strength3}\n"
            
            if prompt and prompt.strip():
                enhanced_prompt += f"\n[ TASK ]: {prompt}"
            
            # 使用CLIP编码
            tokens = clip.tokenize(enhanced_prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            # 构建CONDITIONING格式
            conditioning = [[cond, {"pooled_output": pooled}]]
            
            # 在extra_patches中存储参考信息
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
            
            # 存储主latent信息
            conditioning[0][1]["main_latent_shape"] = list(latent1.shape)
            conditioning[0][1]["reference_count"] = 2
            
            return conditioning
            
        except Exception as e:
            print(f"[Flux Klein Triple] Error encoding prompt: {e}")
            raise
    
    def encode_triple_images(self, clip, vae, image1, image2, image3, 
                           prompt="", strength2=1.0, strength3=1.0):
        """主函数：编码三张图像并生成CONDITIONING"""
        try:
            print(f"[Flux Klein Triple] Starting encoding...")
            print(f"[Flux Klein Triple] Device: {self.device}")
            
            # 编码三张图像
            latent1_main, size1 = self.encode_image_to_latent(vae, image1)
            latent2_ref, size2 = self.encode_image_to_latent(vae, image2)
            latent3_ref, size3 = self.encode_image_to_latent(vae, image3)
            
            print(f"[Flux Klein Triple] Encoded latents - Main: {size1}, Ref1: {size2}, Ref2: {size3}")
            
            # 生成CONDITIONING
            conditioning = self.encode_prompt_with_images(
                clip, prompt, latent1_main, latent2_ref, latent3_ref, strength2, strength3
            )
            
            print(f"[Flux Klein Triple] Encoding completed successfully!")
            return (conditioning,)
            
        except Exception as e:
            print(f"[Flux Klein Triple] Error in main encoding function: {e}")
            import traceback
            traceback.print_exc()
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
