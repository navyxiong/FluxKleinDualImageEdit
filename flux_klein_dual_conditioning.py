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
        except:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Flux Klein Dual] Device: {self.device}")
    
    def get_vae_device(self, vae):
        """
        安全获取VAE设备的方法
        兼容不同的VAE对象类型
        """
        try:
            # 方法1: 尝试直接访问device属性
            if hasattr(vae, 'device'):
                return vae.device
        except:
            pass
        
        try:
            # 方法2: 尝试访问内部VAE模型
            if hasattr(vae, 'vae') and hasattr(vae.vae, 'device'):
                return vae.vae.device
        except:
            pass
        
        try:
            # 方法3: 尝试parameters()方法（标准PyTorch）
            return next(vae.parameters()).device
        except:
            pass
        
        try:
            # 方法4: 尝试访问第一个参数
            for param in vae.parameters():
                return param.device
        except:
            pass
        
        # 方法5: 返回默认设备（最安全）
        print(f"[Flux Klein Dual] [get_vae_device] Could not detect VAE device, using default: {self.device}")
        return self.device
    
    def normalize_image_tensor(self, image_tensor):
        """规范化图像张量，确保是3通道RGB格式"""
        try:
            if image_tensor is None:
                raise ValueError("Input tensor is None")
            
            # 处理批次维度
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor.squeeze(0)
            
            # 确保在CPU上
            if image_tensor.device != torch.device('cpu'):
                image_tensor = image_tensor.cpu()
            
            # 归一化到0-1
            if image_tensor.max() > 1.0:
                image_tensor = torch.clamp(image_tensor, 0, 255) / 255.0
            elif image_tensor.min() < 0.0:
                image_tensor = torch.clamp(image_tensor, 0, 1.0)
            
            # 处理通道
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3)
            elif len(image_tensor.shape) == 3:
                channels = image_tensor.shape[2]
                if channels == 1:
                    image_tensor = image_tensor.repeat(1, 1, 3)
                elif channels == 4:
                    image_tensor = image_tensor[:, :, :3]
                elif channels == 3:
                    pass
                else:
                    raise ValueError(f"Unsupported channel count: {channels}")
            
            # 验证
            if len(image_tensor.shape) != 3 or image_tensor.shape[2] != 3:
                raise ValueError(f"Final shape incorrect: {image_tensor.shape}")
            
            return image_tensor
            
        except Exception as e:
            print(f"[Flux Klein Dual] [normalize] ERROR: {e}")
            return None
    
    def tensor_to_pil(self, image_tensor):
        """将tensor转换为PIL图像"""
        try:
            normalized_tensor = self.normalize_image_tensor(image_tensor)
            if normalized_tensor is None:
                return None
            
            image_tensor = (normalized_tensor * 255).clamp(0, 255).byte()
            image_array = image_tensor.numpy()
            pil_image = Image.fromarray(image_array, 'RGB')
            return pil_image
                
        except Exception as e:
            print(f"[Flux Klein Dual] [to_pil] ERROR: {e}")
            return None
    
    def pad_to_multiple(self, image_tensor, multiple=32, min_size=128):
        """
        使用填充(padding)而不是缩放，避免质量损失
        """
        try:
            # 获取维度
            if len(image_tensor.shape) == 4:
                batch, channels, height, width = image_tensor.shape
            elif len(image_tensor.shape) == 3:
                channels, height, width = image_tensor.shape
                image_tensor = image_tensor.unsqueeze(0)
                batch = 1
            else:
                raise ValueError(f"Unexpected shape: {image_tensor.shape}")
            
            print(f"[Flux Klein Dual] [pad] Original size: {height}x{width}")
            
            # 计算目标尺寸：确保是32的倍数且不小于128
            target_height = max(min_size, ((height + multiple - 1) // multiple) * multiple)
            target_width = max(min_size, ((width + multiple - 1) // multiple) * multiple)
            
            if target_height != height or target_width != width:
                # 计算需要填充的尺寸
                pad_height = target_height - height
                pad_width = target_width - width
                
                # 使用reflect填充避免边缘异常
                padded_tensor = torch.nn.functional.pad(
                    image_tensor,
                    (0, pad_width, 0, pad_height),  # (left, right, top, bottom)
                    mode='reflect'
                )
                
                print(f"[Flux Klein Dual] [pad] Padded: {height}x{width} -> {target_height}x{target_width}")
                
                return padded_tensor, (height, width)
            else:
                print(f"[Flux Klein Dual] [pad] No padding needed")
                return image_tensor, (height, width)
                
        except Exception as e:
            print(f"[Flux Klein Dual] [pad] ERROR: {e}")
            return None, None
    
    def encode_image_to_latent(self, vae, image_tensor):
        """将图像编码为latent，使用填充确保VAE兼容性"""
        try:
            print(f"[Flux Klein Dual] [encode] Starting encoding...")
            
            if image_tensor is None:
                raise ValueError("Input tensor is None")
            
            pil_image = self.tensor_to_pil(image_tensor)
            if pil_image is None:
                raise ValueError("Failed to convert tensor to PIL")
            
            orig_width, orig_height = pil_image.size
            
            # 转换为tensor (B, C, H, W)
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_torch = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
            image_torch = image_torch.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            
            # 关键修复：安全获取VAE设备
            vae_device = self.get_vae_device(vae)
            image_torch = image_torch.to(vae_device)
            
            print(f"[Flux Klein Dual] [encode] Before padding: {image_torch.shape}")
            print(f"[Flux Klein Dual] [encode] Using VAE device: {vae_device}")
            
            # 填充到32的倍数
            image_torch, original_size = self.pad_to_multiple(image_torch, multiple=32, min_size=128)
            
            if image_torch is None:
                raise ValueError("Failed to pad image for VAE")
            
            print(f"[Flux Klein Dual] [encode] After padding: {image_torch.shape}")
            
            # 使用VAE编码
            latent = vae.encode(image_torch)
            
            print(f"[Flux Klein Dual] [encode] SUCCESS - Latent shape: {latent.shape}")
            
            # 返回latent和原始图像尺寸
            return latent, (orig_width, orig_height)
            
        except Exception as e:
            print(f"[Flux Klein Dual] [encode] **FATAL ERROR**: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def encode_prompt_with_images(self, clip, prompt, latent_main, ref_latent=None, ref_strength=1.0):
        """编码提示词并融合图像信息"""
        try:
            enhanced_prompt = f"[MAIN]: Edit image (shape: {list(latent_main.shape)})\n"
            
            if ref_latent is not None:
                enhanced_prompt += f"[REF]: Strength {ref_strength}\n"
            
            if prompt and prompt.strip():
                enhanced_prompt += f"\n[TASK]: {prompt}"
            
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
            
            return conditioning
            
        except Exception as e:
            print(f"[Flux Klein Dual] [prompt] ERROR: {e}")
            raise
    
    def encode_dual_images(self, clip, vae, image1, image2=None, 
                          prompt="", strength=1.0):
        """主函数：编码图像并生成CONDITIONING"""
        try:
            print("\n" + "="*60)
            print("FLUX KLEIN DUAL CONDITIONING - START")
            print("="*60)
            
            # 编码主图像
            print("Step 1: Encoding MAIN image...")
            latent_main, size_main = self.encode_image_to_latent(vae, image1)
            
            if latent_main is None:
                print("\n❌ FATAL: Main image encoding failed")
                print("="*60)
                raise ValueError("Failed to encode main image (image1)")
            
            print(f"✓ SUCCESS: Main latent {latent_main.shape}")
            
            # 编码参考图像
            ref_latent = None
            if image2 is not None:
                print("\nStep 2: Encoding REFERENCE image...")
                ref_latent, _ = self.encode_image_to_latent(vae, image2)
                
                if ref_latent is not None:
                    print(f"✓ SUCCESS: Ref latent {ref_latent.shape}")
                else:
                    print("⚠ Warning: image2 encoding failed, will proceed without it")
            else:
                print("\nStep 2: SKIPPED - No image2")
            
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
            print("✅ ENCODING COMPLETED!")
            print(f"   Main: {latent_main.shape}")
            print(f"   Ref: {'Yes' if ref_latent is not None else 'No'}")
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
    print(f"[Flux Klein Dual] ERROR during registration: {e}")
    import traceback
    traceback.print_exc()
