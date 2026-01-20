# __init__.py
print("[Flux Klein Dual] __init__.py loading...")

try:
    from .flux_klein_dual_conditioning import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    
    print("[Flux Klein Dual] __init__.py loaded successfully!")
    print(f"[Flux Klein Dual] Exported nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
    
except Exception as e:
    print(f"[Flux Klein Dual] Error in __init__.py: {e}")
    import traceback
    traceback.print_exc()
    
    # 即使出错也要提供空映射，避免ComfyUI崩溃
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
