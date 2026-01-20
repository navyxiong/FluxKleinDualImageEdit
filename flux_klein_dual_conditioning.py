# flux_klein_dual_conditioning.py
import torch
import numpy as np
from PIL import Image
import sys
import os

print("[Flux Klein Dual] Loading module...")

# 确保能导入comfy相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import comfy.model_management
    from comfy.model_management import get_torch_device
    print("[Flux Klein Dual] Comfy imports successful!")
except ImportError as e:
    print(f"[Flux Klein Dual] Import error: {e}")

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
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "基于参考图像进行编辑",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
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
            print(f"[Flux Klein Dual] Device initialized: {self.device}")
        except:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[Flux Klein Dual] Fallback device: {self.device}")
    
    def tensor_to_pil(self, image_tensor):
        """将tensor转换为PIL图像"""
        try:
            if image_tensor is None:
                print("[Flux Klein Dual] ERROR: Input tensor is None")
                return None
                
            print(f"[Flux Klein Dual] Converting tensor to PIL - Shape: {image_tensor.shape}, Dtype: {image_tensor.dtype}, Device: {image_tensor.device}, Max: {image_tensor.max()}")
            
            # 处理批次维度
            if len(image_tensor.shape) == 4:
                if image_tensor.shape[0] > 1:
                    print("[Flux Klein Dual] WARNING: Batch size > 1, using first image")
                image_tensor = image_tensor.squeeze(0)
            
            # 确保在CPU上
            if image_tensor.device != torch.device('cpu'):
                image_tensor = image_tensor.cpu()
            
            # 转换0-1范围到0-255
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).clamp(0, 255)
            
            image_tensor = image_tensor.byte()
            image_array = image_tensor.numpy()
            
            print(f"[Flux Klein Dual] Final array shape: {image_array.shape}, dtype: {image_array.dtype}")
            
            # 转换为PIL
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                return Image.fromarray(image_array.astype('uint8'), 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                return Image.fromarray(image_array.astype('uint8'), 'RGBA')
            elif len(image_array.shape) == 2:
                return Image.fromarray(image_array.astype('uint8'), 'L')
            else:
                print(f"[Flux Klein Dual] WARNING: Unusual array shape {image_array.shape}, attempting conversion")
                return Image.fromarray(image_array.astype('uint8'))
                
        except Exception as e:
            print(f"[Flux Klein Dual] ERROR in tensor_to_pil: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def encode_image_to_latent(self, vae, image_tensor):
        """将图像编码为latent"""
        try:
            if image_tensor is None:
                print("[Flux Klein Dual] ERROR: encode_image_to_latent received None tensor")
                return None, None
                
            print(f"[Flux Klein Dual] Encoding image tensor - Shape: {image_tensor.shape}")
            
            pil_image = self.tensor_to_pil(image_tensor)
            if pil_image is None:
                print("[Flux Klein Dual] ERROR: tensor_to_pil returned None")
                return None, None
            
            print(f"[Flux Klein Dual] PIL image size: {pil_image.size}")
            
            width, height = pil_image.size
            
            # 转换为tensor
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_torch = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
            
            print(f"[Flux Klein Dual] Torch tensor shape: {image_torch.shape}")
            
            # 调整维度顺序
            if len(image_torch.shape) == 4 and image_torch.shape[-1] == 3:
                image_torch = image_torch.permute(0, 3, 1, 2)
            
            print(f"[Flux Klein Dual] Final tensor shape for VAE: {image_torch.shape}")
            
            # 使用VAE编码
            latent = vae.encode(image_torch)
            
            print(f"[Flux Klein Dual] VAE encoding successful - Latent shape: {latent.shape}")
            
            return latent, (width, height)
            
        except Exception as e:
            print(f"[Flux Klein Dual] ERROR in encode_image_to_latent: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def encode_prompt_with_images(self, clip, prompt, latent_main, ref_latent=None, ref_strength=1.0):
        """编码提示词并融合图像信息"""
        try:
            print(f"[Flux Klein Dual] Encoding prompt...")
            print(f"[Flux Klein Dual] Main latent shape: {latent_main.shape}")
            print(f"[Flux Klein Dual] Has reference: {ref_latent is not None}")
            
            # 构建增强提示词
            enhanced_prompt = f"[MAIN]: Edit this image (latent shape: {list(latent_main.shape)})\n"
            
            if ref_latent is not None:
                enhanced_prompt += f"[REF]: Reference with strength {ref_strength}\n"
            
            if prompt and prompt.strip():
                enhanced_prompt += f"\n[ TASK ]: {prompt}"
            
            print(f"[Flux Klein Dual] Enhanced prompt: {enhanced_prompt[:100]}...")
            
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
            
            print("[Flux Klein Dual] Prompt encoding successful!")
            return conditioning
            
        except Exception as e:
            print(f"[Flux Klein Dual] ERROR in encode_prompt_with_images: {e}")
            raise
    
    def encode_dual_images(self, clip, vae, image1, image2=None, 
                          prompt="", strength=1.0):
        """主函数：编码图像并生成CONDITIONING"""
        try:
            print("\n" + "="*50)
            print("[Flux Klein Dual] STARTING ENCODING PROCESS")
            print("="*50)
            
            # 编码主图像（必需）
            print(f"[Flux Klein Dual] Step 1: Encoding main image (image1)")
            print(f"[Flux Klein Dual] image1 type: {type(image1)}")
            print(f"[Flux Klein Dual] image1 shape: {image1.shape if hasattr(image1, 'shape') else 'No shape attribute'}")
            
            latent_main, size_main = self.encode_image_to_latent(vae, image1)
            
            if latent_main is None:
                print(f"[Flux Klein Dual] ERROR: Failed to encode main image")
                print(f"[Flux Klein Dual] This usually means:")
                print(f"[Flux Klein Dual] 1. image1 is not a valid tensor")
                print(f"[Flux Klein Dual] 2. image1 has wrong format (not RGB)")
                print(f"[Flux Klein Dual] 3. VAE encoding failed")
                raise ValueError("Failed to encode main image (image1) - See console for details")
            
            print(f"[Flux Klein Dual] Step 1: SUCCESS - Main latent shape: {latent_main.shape}")
            
            # 编码可选的参考图像
            ref_latent = None
            if image2 is not None:
                print(f"[Flux Klein Dual] Step 2: Encoding reference image (image2)")
                print(f"[Flux Klein Dual] image2 type: {type(image2)}")
                print(f"[Flux Klein Dual] image2 shape: {image2.shape if hasattr(image2, 'shape') else 'No shape attribute'}")
                
                ref_latent, _ = self.encode_image_to_latent(vae, image2)
                
                if ref_latent is not None:
                    print(f"[Flux Klein Dual] Step 2: SUCCESS - Reference latent shape: {ref_latent.shape}")
                else:
                    print(f"[Flux Klein Dual] Step 2: SKIPPED - Could not encode image2")
            else:
                print(f"[Flux Klein Dual] Step 2: SKIPPED - No image2 provided")
            
            # 生成CONDITIONING
            print(f"[Flux Klein Dual] Step 3: Encoding prompt")
            conditioning = self.encode_prompt_with_images(
                clip, prompt, latent_main, ref_latent, strength
            )
            
            print(f"[Flux Klein Dual] Step 3: SUCCESS")
            
            # 存储尺寸信息
            conditioning[0][1]["size_info"] = {
                "main": size_main,
                "has_reference": ref_latent is not None
            }
            
            print("\n" + "="*50)
            print(f"[Flux Klein Dual] ENCODING COMPLETED SUCCESSFULLY!")
            print("="*50 + "\n")
            
            return (conditioning,)
            
        except Exception as e:
            print("\n" + "="*50)
            print(f"[Flux Klein Dual] FATAL ERROR: {e}")
            print("="*50)
            import traceback
            traceback.print_exc()
            raise

# 节点注册
try:
    NODE_CLASS_MAPPINGS = {
        "FluxKleinDualConditioning": FluxKleinDualConditioning,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "FluxKleinDualConditioning": "Flux Klein Dual Conditioning",
    }

    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    
    print("[Flux Klein Dual] Node class mappings registered successfully!")
    print(f"[Flux Klein Dual] Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
    
except Exception as e:
    print(f"[Flux Klein Dual] ERROR during node registration: {e}")
    import traceback
    traceback.print_exc()
