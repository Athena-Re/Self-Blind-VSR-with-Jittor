import os
import sys
import tempfile
from .correlation_pytorch import FunctionCorrelationPytorch, ModuleCorrelationPytorch

# 检查是否强制使用PyTorch实现
force_pytorch = False  # 默认尝试使用CUDA实现
if 'FORCE_CORRELATION_PYTORCH' in os.environ:
    force_pytorch = os.environ['FORCE_CORRELATION_PYTORCH'].lower() in ('true', '1', 't')

# 检查临时目录设置
temp_dir = os.environ.get('TEMP', None) or os.environ.get('TMP', None) or os.environ.get('TMPDIR', None)
if temp_dir and os.path.exists(temp_dir):
    print(f"✓ 使用临时目录: {temp_dir}")
    # 确保临时目录不包含中文字符
    has_non_ascii = any(ord(c) > 127 for c in temp_dir)
    if has_non_ascii:
        print(f"⚠️ 警告: 临时目录包含非ASCII字符，可能导致CUDA编译失败")
        
        # 尝试使用系统临时目录
        system_temp = tempfile.gettempdir()
        if not any(ord(c) > 127 for c in system_temp):
            print(f"✓ 使用系统临时目录: {system_temp}")
            os.environ['TEMP'] = system_temp
            os.environ['TMP'] = system_temp
            os.environ['TMPDIR'] = system_temp
            temp_dir = system_temp
else:
    print("⚠️ 警告: 未找到有效的临时目录")

# 导入CUDA实现（如果不强制使用PyTorch实现）
if not force_pytorch:
    try:
        # 保存原始临时目录环境变量
        orig_tmp = os.environ.get('TMPDIR')
        orig_temp = os.environ.get('TEMP')
        orig_tmp_env = os.environ.get('TMP')
        
        # 尝试使用D盘临时目录
        cuda_temp_dir = 'D:\\TEMP\\cuda_cache'
        if os.path.exists(cuda_temp_dir):
            os.environ['TMPDIR'] = cuda_temp_dir
            os.environ['TEMP'] = cuda_temp_dir
            os.environ['TMP'] = cuda_temp_dir
            print(f"✓ 临时设置CUDA编译临时目录: {cuda_temp_dir}")
        
        # 尝试导入CUDA实现
        from .correlation import FunctionCorrelation, ModuleCorrelation
        print("✅ CUDA correlation实现已成功加载")
        
    except Exception as e:
        # 恢复原始环境变量
        if orig_tmp is not None:
            os.environ['TMPDIR'] = orig_tmp
        if orig_temp is not None:
            os.environ['TEMP'] = orig_temp
        if orig_tmp_env is not None:
            os.environ['TMP'] = orig_tmp_env
            
        print("⚠️ CUDA correlation加载失败，自动切换到PyTorch实现")
        error_msg = str(e)
        # 打印详细错误信息
        print(f"  错误详情: {error_msg}")
        
        # 检查是否是中文路径问题
        if "cannot open source file" in error_msg and any(ord(c) > 127 for c in error_msg):
            print("  ⚠️ 检测到中文路径问题，请确保临时目录不包含中文字符")
            print("  💡 提示: 当前已自动切换到PyTorch实现，不影响功能，但可能会略微降低性能")
        
        # 使用PyTorch实现作为回退
        from .correlation_pytorch import FunctionCorrelationPytorch as FunctionCorrelation
        from .correlation_pytorch import ModuleCorrelationPytorch as ModuleCorrelation
        print("  ✓ 已使用PyTorch correlation实现")
else:
    # 强制使用PyTorch实现
    from .correlation_pytorch import FunctionCorrelationPytorch as FunctionCorrelation
    from .correlation_pytorch import ModuleCorrelationPytorch as ModuleCorrelation
    print("⚠️ 已设置强制使用PyTorch correlation实现")

__all__ = ['FunctionCorrelation', 'ModuleCorrelation', 'FunctionCorrelationPytorch', 'ModuleCorrelationPytorch'] 