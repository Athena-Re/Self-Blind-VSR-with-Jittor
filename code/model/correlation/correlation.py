import torch
import os
import tempfile

import cupy
import re

# 设置cupy编译的临时目录
def setup_cupy_temp_dir():
    """设置cupy编译使用的临时目录"""
    temp_dir = 'D:\\TEMP'
    
    # 确保目录存在
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    
    # 设置cupy相关的环境变量
    os.environ['CUPY_CACHE_DIR'] = os.path.join(temp_dir, 'cupy_cache')
    os.environ['CUPY_DUMP_CUDA_SOURCE_ON_ERROR'] = '0'  # 减少错误时的输出
    
    # 创建cupy缓存目录
    if not os.path.exists(os.environ['CUPY_CACHE_DIR']):
        os.makedirs(os.environ['CUPY_CACHE_DIR'], exist_ok=True)
    
    # 设置临时目录环境变量
    orig_temp = os.environ.get('TEMP')
    orig_tmp = os.environ.get('TMP')
    orig_tmpdir = os.environ.get('TMPDIR')
    
    os.environ['TEMP'] = temp_dir
    os.environ['TMP'] = temp_dir  
    os.environ['TMPDIR'] = temp_dir
    
    print(f"✓ 已设置cupy编译临时目录: {temp_dir}")
    
    return orig_temp, orig_tmp, orig_tmpdir

def restore_temp_env(orig_temp, orig_tmp, orig_tmpdir):
    """恢复原始临时目录环境变量"""
    if orig_temp is not None:
        os.environ['TEMP'] = orig_temp
    if orig_tmp is not None:
        os.environ['TMP'] = orig_tmp  
    if orig_tmpdir is not None:
        os.environ['TMPDIR'] = orig_tmpdir

# 在导入时设置临时目录
orig_env = setup_cupy_temp_dir()


class Stream:
    ptr = torch.cuda.current_stream().cuda_stream


# end

kernel_Correlation_rearrange = '''
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	  if (intIndex >= n) {
	    return;
	  }

	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;

	  float dblValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

	  __syncthreads();

	  int intPaddedY = (intIndex / SIZE_3(input)) + 4;
	  int intPaddedX = (intIndex % SIZE_3(input)) + 4;
	  int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;

	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = dblValue;
	}
'''

kernel_Correlation_updateOutput = '''
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
	  extern __shared__ char patch_data_char[];
	  
	  float *patch_data = (float *)patch_data_char;
	  
	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = blockIdx.x + 4;
	  int y1 = blockIdx.y + 4;
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;
	  
	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = (j + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }
	  
	  __syncthreads();
	  
	  __shared__ float sum[32];
	  
	  // Compute correlation
	  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;
	  
	    int s2o = top_channel % 9 - 4;
	    int s2p = top_channel / 9 - 4;
	    
	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;
	          
	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;
	          
	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }
	    
	    __syncthreads();
	    
	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  }
	}
'''

kernel_Correlation_updateGradFirst = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradFirst(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradFirst); // channels
	  int l = (intIndex / SIZE_1(gradFirst)) % SIZE_3(gradFirst) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradFirst) / SIZE_3(gradFirst)) % SIZE_2(gradFirst) + 4; // h-pos
	  
	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;
	  
	  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	  int xmin = (l - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	  int ymin = (m - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	  
	  // Same here:
	  int xmax = (l - 4 + round_off_s1) - round_off; // floor (l - 4)
	  int ymax = (m - 4 + round_off_s1) - round_off; // floor (m - 4)
	  
	  float sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	    xmin = max(0,xmin);
	    xmax = min(SIZE_3(gradOutput)-1,xmax);
	    
	    ymin = max(0,ymin);
	    ymax = min(SIZE_2(gradOutput)-1,ymax);
	    
	    for (int p = -4; p <= 4; p++) {
	      for (int o = -4; o <= 4; o++) {
	        // Get rbot1 data:
	        int s2o = o;
	        int s2p = p;
	        int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
	        float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]
	        
	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
	        
	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradFirst);
	  const int bot0index = ((n * SIZE_2(gradFirst)) + (m-4)) * SIZE_3(gradFirst) + (l-4);
	  gradFirst[bot0index + intSample*SIZE_1(gradFirst)*SIZE_2(gradFirst)*SIZE_3(gradFirst)] = sum / (float)sumelems;
	} }
'''

kernel_Correlation_updateGradSecond = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradSecond(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradSecond); // channels
	  int l = (intIndex / SIZE_1(gradSecond)) % SIZE_3(gradSecond) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradSecond) / SIZE_3(gradSecond)) % SIZE_2(gradSecond) + 4; // h-pos
	  
	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;
	  
	  float sum = 0;
	  for (int p = -4; p <= 4; p++) {
	    for (int o = -4; o <= 4; o++) {
	      int s2o = o;
	      int s2p = p;
	      
	      int x_im = l + s2o;
	      int y_im = m + s2p;
	      int op = (p+4) * 9 + (o+4); // index[o,p]
	      
	      if (y_im >= (0 + 4) && y_im < (SIZE_2(gradSecond) + 4) && x_im >= (0 + 4) && x_im < (SIZE_3(gradSecond) + 4)) {
	        int idxbot0 = ((intSample * SIZE_1(rbot0) + (y_im)) * SIZE_2(rbot0) + (x_im)) * SIZE_3(rbot0) + n;
	        float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]
	        
	        const int round_off = ROUND_OFF;
	        const int round_off_s1 = round_off;
	        
	        int xmin = (x_im - s2o - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	        int ymin = (y_im - s2p - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	        
	        // Same here:
	        int xmax = (x_im - s2o - 4 + round_off_s1) - round_off; // floor (l - 4)
	        int ymax = (y_im - s2p - 4 + round_off_s1) - round_off; // floor (m - 4)
	        
	        if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	          xmin = max(0,xmin);
	          xmax = min(SIZE_3(gradOutput)-1,xmax);
	          
	          ymin = max(0,ymin);
	          ymax = min(SIZE_2(gradOutput)-1,ymax);
	          
	          // Index offset for gradOutput in following loops:
	          int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
	          
	          for (int y = ymin; y <= ymax; y++) {
	            for (int x = xmin; x <= xmax; x++) {
	              int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	              sum += gradOutput[idxgradOutput] * bot0tmp;
	            }
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradSecond);
	  const int bot1index = ((n * SIZE_2(gradSecond)) + (m-4)) * SIZE_3(gradSecond) + (l-4);
	  gradSecond[bot1index + intSample*SIZE_1(gradSecond)*SIZE_2(gradSecond)*SIZE_3(gradSecond)] = sum / (float)sumelems;
	} }
'''


# end

def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objectMatch.group(0), '(' + str.join('+', strIndex) + ')')
    # end

    return strKernel


# end

# Global kernel cache
_kernel_cache = {}

def cupy_launch(strFunction, strKernel):
    # Create a cache key based on function name and kernel
    cache_key = (strFunction, strKernel)
    
    # Check if kernel is already compiled and cached
    if cache_key not in _kernel_cache:
        try:
            # 在编译前重新设置临时目录
            temp_backup = setup_cupy_temp_dir()
            
            # Try different cupy compilation methods based on version
            if hasattr(cupy, 'RawKernel'):
                # Modern cupy version (>= 8.0)
                kernel_obj = cupy.RawKernel(strKernel, strFunction)
                _kernel_cache[cache_key] = kernel_obj
            elif hasattr(cupy.cuda, 'compile_with_cache'):
                # Older cupy version with compile_with_cache
                _kernel_cache[cache_key] = cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
            elif hasattr(cupy, 'cuda') and hasattr(cupy.cuda, 'compile'):
                # Alternative compilation method
                _kernel_cache[cache_key] = cupy.cuda.compile(strKernel).get_function(strFunction)
            else:
                raise RuntimeError("Unable to compile CUDA kernel with current cupy version")
                
            print(f"✓ CUDA kernel '{strFunction}' 编译成功")
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ CUDA kernel '{strFunction}' 编译失败: {error_msg}")
            
            # 检查是否是中文路径问题
            if "cannot open source file" in error_msg and any(ord(c) > 127 for c in error_msg):
                print("  💡 检测到中文路径问题，自动切换到PyTorch实现")
            
            # Create a dummy function that raises an error when called
            def dummy_kernel(*args, **kwargs):
                raise RuntimeError(f"CUDA kernel compilation failed: {e}")
            _kernel_cache[cache_key] = dummy_kernel
        finally:
            # 恢复环境变量
            restore_temp_env(*temp_backup)
    
    return _kernel_cache[cache_key]


# end

class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(self, first, second):
        rbot0 = first.new_zeros([first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)])
        rbot1 = first.new_zeros([first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)])

        self.save_for_backward(first, second, rbot0, rbot1)

        assert (first.is_contiguous() == True)
        assert (second.is_contiguous() == True)

        output = first.new_zeros([first.size(0), 81, first.size(2), first.size(3)])

        if first.is_cuda == True:
            n = first.size(2) * first.size(3)
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'input': first,
                'output': rbot0
            }))(
                grid=tuple([int((n + 16 - 1) / 16), first.size(1), first.size(0)]),
                block=tuple([16, 1, 1]),
                args=[n, first.data_ptr(), rbot0.data_ptr()],
                stream=Stream
            )

            n = second.size(2) * second.size(3)
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'input': second,
                'output': rbot1
            }))(
                grid=tuple([int((n + 16 - 1) / 16), second.size(1), second.size(0)]),
                block=tuple([16, 1, 1]),
                args=[n, second.data_ptr(), rbot1.data_ptr()],
                stream=Stream
            )

            n = output.size(1) * output.size(2) * output.size(3)
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
                'rbot0': rbot0,
                'rbot1': rbot1,
                'top': output
            }))(
                grid=tuple([output.size(3), output.size(2), output.size(0)]),
                block=tuple([32, 1, 1]),
                shared_mem=first.size(1) * 4,
                args=[n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        elif first.is_cuda == False:
            raise NotImplementedError()

        # end

        return output

    # end

    @staticmethod
    def backward(self, gradOutput):
        first, second, rbot0, rbot1 = self.saved_tensors

        assert (gradOutput.is_contiguous() == True)

        gradFirst = first.new_zeros([first.size(0), first.size(1), first.size(2), first.size(3)]) if \
        self.needs_input_grad[0] == True else None
        gradSecond = first.new_zeros([first.size(0), first.size(1), first.size(2), first.size(3)]) if \
        self.needs_input_grad[1] == True else None

        if first.is_cuda == True:
            if gradFirst is not None:
                for intSample in range(first.size(0)):
                    n = first.size(1) * first.size(2) * first.size(3)
                    cupy_launch('kernel_Correlation_updateGradFirst',
                                cupy_kernel('kernel_Correlation_updateGradFirst', {
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': gradOutput,
                                    'gradFirst': gradFirst,
                                    'gradSecond': None
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(),
                              gradFirst.data_ptr(), None],
                        stream=Stream
                    )
            # end
            # end

            if gradSecond is not None:
                for intSample in range(first.size(0)):
                    n = first.size(1) * first.size(2) * first.size(3)
                    cupy_launch('kernel_Correlation_updateGradSecond',
                                cupy_kernel('kernel_Correlation_updateGradSecond', {
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': gradOutput,
                                    'gradFirst': None,
                                    'gradSecond': gradSecond
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), None,
                              gradSecond.data_ptr()],
                        stream=Stream
                    )
            # end
        # end

        elif first.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradFirst, gradSecond
# end


# end

# Global flag to track if CUDA implementation is available
_cuda_available = None

def _check_cuda_availability():
    global _cuda_available
    if _cuda_available is None:
        try:
            # Try to compile a simple kernel to test CUDA availability
            test_tensor = torch.zeros(1, 1, 8, 8, device='cuda', dtype=torch.float32)
            _ = _FunctionCorrelation.apply(test_tensor, test_tensor)
            _cuda_available = True
            print("✅ CUDA correlation implementation available")
        except Exception as e:
            _cuda_available = False
            print(f"⚠️ CUDA correlation compilation failed, using PyTorch fallback: {e}")
    return _cuda_available

def FunctionCorrelation(tensorFirst, tensorSecond):
    if tensorFirst.is_cuda and _check_cuda_availability():
        try:
            return _FunctionCorrelation.apply(tensorFirst, tensorSecond)
        except Exception as e:
            print(f"⚠️ CUDA correlation failed, falling back to PyTorch implementation: {e}")
            # Import PyTorch fallback
            from .correlation_pytorch import FunctionCorrelationPytorch
            return FunctionCorrelationPytorch(tensorFirst, tensorSecond)
    else:
        # Use PyTorch fallback for CPU or when CUDA is not available
        from .correlation_pytorch import FunctionCorrelationPytorch
        return FunctionCorrelationPytorch(tensorFirst, tensorSecond)


# end

class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    # end

    def forward(self, tensorFirst, tensorSecond):
        return _FunctionCorrelation.apply(tensorFirst, tensorSecond)
# end
# end
