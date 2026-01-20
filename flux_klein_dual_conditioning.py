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
    自动调整尺寸为32的倍数
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
        except:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Flux Klein Dual] Device: {self.device}")
    
    def normalize_image_tensor(self, image_tensor):
        """
        规范化图像张量，确保是3通道RGB格式
        """
        try:
            if image_tensor is None:
                raise ValueError("Input tensor is None")
            
            print(f"[Flux Klein Dual] [normalize] Input shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            
            # 处理批次维度
            if len(image_tensor.shape) == 4:
                if image_tensor.shape[0] > 1:
                    print(f"[Flux Klein Dual] [normalize] Warning: Batch size > 1, using first image")
                image_tensor = image_tensor.squeeze(0)
            
            # 确保在CPU上便于处理
            if image_tensor.device != torch.device('cpu'):
                image_tensor = image_tensor.cpu()
            
            # 确保数值范围在0-1
            if image_tensor.max() > 1.0:
                print(f"[Flux Klein Dual] [normalize] Warning: Values > 1.0 detected, normalizing")
                image_tensor = torch.clamp(image_tensor, 0, 255) / 255.0
            elif image_tensor.min() < 0.0:
                print(f"[Flux Klein Dual] [normalize] Warning: Negative values detected, clamping")
                image_tensor = torch.clamp(image_tensor, 0, 1.0)
            
            # 处理通道维度
            if len(image_tensor.shape) == 2:
                # 灰度图 -> RGB
                print(f"[Flux Klein Dual] [normalize] Converting grayscale to RGB")
                image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3)
            elif len(image_tensor.shape) == 3:
                channels = image_tensor.shape[2]
                if channels == 1:
                    # 单通道 -> RGB
                    print(f"[Flux Klein Dual] [normalize] Converting single channel to RGB")
                    image_tensor = image_tensor.repeat(1, 1, 3)
                elif channels == 4:
                    # RGBA -> RGB (移除alpha通道)
                    print(f"[Flux Klein Dual] [normalize] Converting RGBA to RGB")
                    image_tensor = image_tensor[:, :, :3]
                elif channels == 3:
                    # 已经是RGB，无需处理
                    pass
                else:
                    raise ValueError(f"Unsupported channel count: {channels}")
            else:
                raise ValueError(f"Unsupported tensor dimensions: {len(image_tensor.shape)}")
            
            # 最终验证
            if len(image_tensor.shape) != 3 or image_tensor.shape[2] != 3:
                raise ValueError(f"Final tensor shape is incorrect: {image_tensor.shape}")
            
            return image_tensor
            
        except Exception as e:
            print(f"[Flux Klein Dual] [normalize] ERROR: {e}")
            return None
    
    def tensor_to_pil(self, image_tensor):
        """将tensor转换为PIL图像（使用规范化后的张量）"""
        try:
            print(f"[Flux Klein Dual] [to_pil] Starting conversion...")
            
            # 先规范化张量
            normalized_tensor = self.normalize_image_tensor(image_tensor)
            if normalized_tensor is None:
                return None
            
            # 转换到0-255范围
            image_tensor = (normalized_tensor * 255).clamp(0, 255).byte()
            
            # 转换为numpy
            image_array = image_tensor.numpy()
            
            # 转换为PIL
            pil_image = Image.fromarray(image_array, 'RGB')
            print(f"[Flux Klein Dual] [to_pil] Success - PIL size: {pil_image.size}")
            return pil_image
                
        except Exception as e:
            print(f"[Flux Klein Dual] [to_pil] ERROR: {e}")
            return None
    
    def resize_for_vae(self, image_tensor):
        """
        关键修复：调整图像尺寸为32的倍数
        这是解决VAE编码失败的必要步骤
        """
        try:
            print(f"[Flux Klein Dual] [resize] Checking VAE compatibility...")
            
            # 假设输入是(B, C, H, W)或(C, H, W)格式
            if len(image_tensor.shape) == 4:
                batch, channels, height, width = image_tensor.shape
            elif len(image_tensor.shape) == 3:
                channels, height, width = image_tensor.shape
                image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
                batch = 1
            else:
                raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")
            
            print(f"[Flux Klein Dual] [resize] Original size: {height}x{width}")
            
            # 计算32的倍数尺寸（向上取整）
            new_height = ((height + 31) // 32) * 32
            new_width = ((width + 31) // 32) * 32
            
            if new_height != height or new_width != width:
                print(f"[Flux Klein Dual] [resize] Resizing for VAE: {height}x{width} -> {new_height}x{new_width}")
                resized_tensor = torch.nn.functional.interpolate(
                    image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False
                )
                return resized_tensor, (height, width)  # 返回新张量和原始尺寸
            else:
                print(f"[Flux Klein Dual] [resize] Size is already VAE-compatible")
                return image_tensor, (height, width)
                
        except Exception as e:
            print(f"[Flux Klein Dual] [resize] ERROR: {e}")
            return None, None
    
    def encode_image_to_latent(self, vae, image_tensor):
        """将图像编码为latent，自动调整尺寸以满足VAE要求"""
        try:
            print(f"[Flux Klein Dual] [encode] Encoding image to latent...")
            
            if image_tensor is None:
                raise ValueError("Input image_tensor is None")
            
            pil_image = self.tensor_to_pil(image_tensor)
            if pil_image is None:
                raise ValueError("Failed to convert tensor to PIL")
            
            orig_width, orig_height = pil_image.size
            
            # 转换为tensor
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_torch = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
            
            # 调整维度顺序
            if len(image_torch.shape) == 4 and image_torch.shape[-1] == 3:
                image_torch = image_torch.permute(0, 3, 1, 2)
            
            # 确保VAE在正确设备上
            vae_device = next(vae.parameters()).device
            image_torch = image_torch.to(vae_device)
            
            # 关键修复：确保尺寸是32的倍数
            image_torch, original_size = self.resize_for_vae(image_torch)
            if image_torch is None:
                raise ValueError("Failed to resize image for VAE")
            
            print(f"[Flux Klein Dual] [encode] Final VAE input shape: {image_torch.shape}")
            
            # 使用VAE编码
            latent = vae.encode(image_torch)
            
            print(f"[Flux Klein Dual] [encode] SUCCESS - Latent shape: {latent.shape}")
            
            # 返回latent和原始图像尺寸（用于后续解码）
            return latent, (orig_width, orig_height)
            
        except Exception as e:
            print(f"[Flux Klein Dual] [encode] **FATAL ERROR**: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def encode_prompt_with_images(self, clip, prompt, latent_main, ref_latent=None, ref_strength=1.0):
        """编码提示词并融合图像信息"""
        try:
            print(f"[Flux Klein Dual] [prompt] Encoding prompt...")
            print(f"[Flux Klein Dual] [prompt] Main latent shape: {latent_main.shape}")
            print(f"[Flux Klein Dual] [prompt] Has reference: {ref_latent is not None}")
            
            enhanced_prompt = f"[MAIN]: Edit this image (latent shape: {list(latent_main.shape)})\n"
            
            if ref_latent is not None:
                enhanced_prompt += f"[REF]: Reference with strength {ref_strength}\n"
            
            if prompt and prompt.strip():
                enhanced_prompt += f"\n[ TASK ]: {prompt}"
            
            tokens = clip.tokenize(enhanced_prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            conditioning = [[cond, {"pooled_output": pooled}]]
            
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
            
            conditioning[0][1]["main_latent_shape"] = list(latent_main.shape)
            
            print(f"[Flux Klein Dual] [prompt] SUCCESS")
            return conditioning
            
        except Exception as e:
            print(f"[Flux Klein Dual] [prompt] ERROR: {e}")
            raise
    
    def encode_dual_images(self, clip, vae, image1, image2=None, 
                          prompt="", strength=1.0):
        """主函数：编码图像并生成CONDITIONING"""
        try:
            print("\n" + "="*60)
            print("FLUX KLEIN DUAL CONDITIONING - EXECUTION START")
            print("="*60)
            
            # 编码主图像
            print("Step 1: Encoding MAIN image (image1)...")
            print(f"  image1 shape: {image1.shape if hasattr(image1, 'shape') else 'No shape'}")
            
            latent_main, size_main = self.encode_image_to_latent(vae, image1)
            
            if latent_main is None:
                print("\n❌ FATAL: Failed to encode main image")
                print("="*60)
                raise ValueError("Failed to encode main image (image1) - See logs above")
            
            print(f"✓ SUCCESS: Main latent shape: {latent_main.shape}")
            
            # 编码参考图像
            ref_latent = None
            if image2 is not None:
                print("\nStep 2: Encoding REFERENCE image (image2)...")
                print(f"  image2 shape: {image2.shape if hasattr(image2, 'shape') else 'No shape'}")
                
                ref_latent, _ = self.encode_image_to_latent(vae, image2)
                
                if ref_latent is not None:
                    print(f"✓ SUCCESS: Reference latent shape: {ref_latent.shape}")
                else:
                    print("⚠ Warning: Failed to encode image2, will proceed without reference")
            else:
                print("\nStep 2: SKIPPED - No image2 provided")
            
            # 生成CONDITIONING
            print("\nStep 3: Encoding prompt...")
            conditioning = self.encode_prompt_with_images(
                clip, prompt, latent_main, ref_latent, strength
            )
            
            # 存储尺寸信息
            conditioning[0][1]["size_info"] = {
                "main": size_main,
                "has_reference": ref_latent is not None
            }
            
            print("\n" + "="*60)
            print("✅ ENCODING COMPLETED SUCCESSFULLY!")
            print(f"   - Main latent: {latent_main.shape}")
            print(f"   - Reference: {'Yes' if ref_latent is not None else 'No'}")
            print(f"   - Strength: {strength if ref_latent is not None else 'N/A'}")
            print("="*60 + "\n")
            
            return (conditioning,)
            
        except Exception as e:
            print("\n" + "="*60)
            print(f"❌ FATAL ERROR: {e}")
            print("="*60)
            import traceback
            traceback.print_exc()
            print("="*60 + "\n")
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
    
    print("[Flux Klein Dual] Node registered successfully!")
    
except Exception as e:
    print(f"[Flux Klein Dual] ERROR during node registration: {e}")
    import traceback
    traceback.print_exc()
