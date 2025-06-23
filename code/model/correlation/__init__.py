import os
from .correlation_pytorch import FunctionCorrelationPytorch, ModuleCorrelationPytorch

# 检查是否强制使用PyTorch实现
force_pytorch = False  # 默认尝试使用CUDA实现
if 'FORCE_CORRELATION_PYTORCH' in os.environ:
    force_pytorch = os.environ['FORCE_CORRELATION_PYTORCH'].lower() in ('true', '1', 't')

# 导入CUDA实现（如果不强制使用PyTorch实现）
if not force_pytorch:
    try:
        from .correlation import FunctionCorrelation, ModuleCorrelation
        print("✅ CUDA correlation实现已成功加载")
    except Exception as e:
        print("⚠️ CUDA correlation加载失败，自动切换到PyTorch实现")
        error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
        print("  原因:", error_msg)
        from .correlation_pytorch import FunctionCorrelationPytorch as FunctionCorrelation
        from .correlation_pytorch import ModuleCorrelationPytorch as ModuleCorrelation
        print("  已使用PyTorch correlation实现")
else:
    # 强制使用PyTorch实现
    from .correlation_pytorch import FunctionCorrelationPytorch as FunctionCorrelation
    from .correlation_pytorch import ModuleCorrelationPytorch as ModuleCorrelation
    print("⚠️ 已设置强制使用PyTorch correlation实现")

__all__ = ['FunctionCorrelation', 'ModuleCorrelation', 'FunctionCorrelationPytorch', 'ModuleCorrelationPytorch'] 