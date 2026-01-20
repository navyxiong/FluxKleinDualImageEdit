# flux_klein_dual_conditioning.py
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
    print(f"[Flux Klein Dual] Import error: {e}")
    print("[Flux Klein Dual] Please make sure this file is in ComfyUI/custom_nodes/flux_klein_dual/")

class FluxKleinDualConditioning:
    """
    FLUX.2 [klein] 双图条件编码节点
    输入clip、vae、image1（必需）、image2（可选）
    仅输出CONDITIONING
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image1": ("IMAGE",),  # 主图像 - 决定输出尺寸
            },
            "optional": {
                "image2": ("IMAGE",),  # 参考图像（可选）
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "基于参考图像进行编辑",
                    "tooltip": "编辑提示词，描述如何使用参考图像"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "image2对生成结果的影响强度"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_dual_images"
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
                return None
                
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor.squeeze(0)
            
            if image_tensor.device != torch.device('cpu'):
                image_tensor = image_tensor.cpu()
            
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).clamp(0, 255)
            
            image_tensor = image_tensor.byte()
            image_array = image_tensor.numpy()
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                return Image.fromarray(image_array.astype('uint8'), 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                return Image.fromarray(image_array.astype('uint8'), 'RGBA')
            elif len(image_array.shape) == 2:
                return Image.fromarray(image_array.astype('uint8'), 'L')
            else:
                return Image.fromarray(image_array.astype('uint8'))
                
        except Exception as e:
            print(f"[Flux Klein Dual] Error converting tensor to PIL: {e}")
            return None
    
    def encode_image_to_latent(self, vae, image_tensor):
        """将图像编码为latent"""
        try:
            if image_tensor is None:
                return None, None
                
            pil_image = self.tensor_to_pil(image_tensor)
            if pil_image is None:
                return None, None
            
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
            print(f"[Flux Klein Dual] Error encoding image to latent: {e}")
            return None, None
    
    def encode_prompt_with_images(self, clip, prompt, latent_main, ref_latent=None, ref_strength=1.0):
        """编码提示词并融合图像信息"""
        try:
            # 构建增强提示词
            enhanced_prompt = f"[MAIN]: Edit this image (latent shape: {list(latent_main.shape)})\n"
            
            if ref_latent is not None:
                enhanced_prompt += f"[REF]: Reference with strength {ref_strength}\n"
            
            if prompt and prompt.strip():
                enhanced_prompt += f"\n[ TASK ]: {prompt}"
            
            # 使用CLIP编码
            tokens = clip.tokenize(enhanced_prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            # 构建CONDITIONING格式
            conditioning = [[cond, {"pooled_output": pooled}]]
            
            # 如果有参考图像，添加信息到extra_patches
            if ref_latent is not None:
                conditioning[0][1]["extra_patches"] = {
                    "ref_latent": {
                        "samples": ref_latent,
                        "strength": ref_strength
                    }
                }
                conditioning[0][1]["reference_count"] = 1
            else:
                conditioning[0][1]["reference_count"] = 0
            
            # 存储主latent信息
            conditioning[0][1]["main_latent_shape"] = list(latent_main.shape)
            
            return conditioning
            
        except Exception as e:
            print(f"[Flux Klein Dual] Error encoding prompt: {e}")
            raise
    
    def encode_dual_images(self, clip, vae, image1, image2=None, 
                          prompt="", strength=1.0):
        """主函数：编码图像并生成CONDITIONING"""
        try:
            print(f"[Flux Klein Dual] Starting encoding...")
            print(f"[Flux Klein Dual] Device: {self.device}")
            
            # 编码主图像（必需）
            latent_main, size_main = self.encode_image_to_latent(vae, image1)
            if latent_main is None:
                raise ValueError("Failed to encode main image (image1)")
            
            # 编码可选的参考图像
            ref_latent = None
            if image2 is not None:
                ref_latent, size_ref = self.encode_image_to_latent(vae, image2)
                if ref_latent is not None:
                    print(f"[Flux Klein Dual] Reference image encoded")
            
            # 生成CONDITIONING
            conditioning = self.encode_prompt_with_images(
                clip, prompt, latent_main, ref_latent, strength
            )
            
            # 存储尺寸信息
            conditioning[0][1]["size_info"] = {
                "main": size_main,
                "has_reference": ref_latent is not None
            }
            
            ref_status = "with reference" if ref_latent is not None else "no reference"
            print(f"[Flux Klein Dual] Encoding completed {ref_status}!")
            return (conditioning,)
            
        except Exception as e:
            print(f"[Flux Klein Dual] Error in main encoding function: {e}")
            import traceback
            traceback.print_exc()
            raise

# 节点注册
NODE_CLASS_MAPPINGS = {
    "FluxKleinDualConditioning": FluxKleinDualConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKleinDualConditioning": "Flux Klein Dual Conditioning",
}

# 确保模块级别的导入正确
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 打印加载信息
print("[Flux Klein Dual Conditioning] Node module loaded successfully!")
