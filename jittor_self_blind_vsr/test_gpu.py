#!/usr/bin/env python3
import jittor as jt
import time

def test_gpu():
    print("=== Jittor GPU 测试 ===")
    
    # 检查CUDA可用性
    print(f"1. CUDA 可用性: {jt.has_cuda}")
    
    if not jt.has_cuda:
        print("❌ 系统不支持CUDA，无法使用GPU")
        return False
    
    # 设置使用CUDA
    jt.flags.use_cuda = 1
    print(f"2. CUDA 已启用: {jt.flags.use_cuda}")
    print(f"3. GPU 数量: {jt.get_device_count()}")
    
    try:
        # 创建测试张量
        print("\n--- 执行GPU计算测试 ---")
        
        # 测试基本运算
        a = jt.randn(1000, 1000)
        b = jt.randn(1000, 1000)
        
        # 计时GPU运算
        start_time = time.time()
        for i in range(10):
            c = jt.matmul(a, b)
            jt.sync_all()  # 确保计算完成
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU矩阵运算成功")
        print(f"✅ 10次矩阵乘法耗时: {gpu_time:.4f}s")
        print(f"✅ 结果形状: {c.shape}")
        print(f"✅ 结果样本: {c[0, 0].item():.4f}")
        
        # 测试神经网络层
        print("\n--- 测试神经网络层 ---")
        linear = jt.nn.Linear(100, 50)
        x = jt.randn(32, 100)
        y = linear(x)
        
        print(f"✅ 线性层计算成功")
        print(f"✅ 输入形状: {x.shape}")
        print(f"✅ 输出形状: {y.shape}")
        
        # 测试梯度计算
        print("\n--- 测试梯度计算 ---")
        x = jt.randn(10, 10)
        x.requires_grad = True
        y = jt.sum(x * x)
        
        grad = jt.grad(y, x)
        print(f"✅ 梯度计算成功")
        print(f"✅ 梯度形状: {grad.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def compare_cpu_gpu():
    print("\n=== CPU vs GPU 性能对比 ===")
    
    # CPU测试
    jt.flags.use_cuda = 0
    a = jt.randn(2000, 2000)
    b = jt.randn(2000, 2000)
    
    start_time = time.time()
    for i in range(5):
        c = jt.matmul(a, b)
        jt.sync_all()
    cpu_time = time.time() - start_time
    print(f"🐌 CPU时间 (5次矩阵乘法): {cpu_time:.4f}s")
    
    # GPU测试
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        a = jt.randn(2000, 2000)
        b = jt.randn(2000, 2000)
        
        start_time = time.time()
        for i in range(5):
            c = jt.matmul(a, b)
            jt.sync_all()
        gpu_time = time.time() - start_time
        print(f"🚀 GPU时间 (5次矩阵乘法): {gpu_time:.4f}s")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"⚡ 加速比: {speedup:.2f}x")

if __name__ == "__main__":
    success = test_gpu()
    if success:
        compare_cpu_gpu()
        print("\n🎉 GPU测试完成！可以正常使用GPU训练。")
    else:
        print("\n⚠️  建议使用CPU进行训练，或检查CUDA安装。") 