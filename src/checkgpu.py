# 导入PyTorch库
import torch
# 导入PyTorch的cuDNN后端模块
import torch.backends.cudnn as cudnn


# 定义一个检查GPU状态的函数
def check_gpu():
    # 检查是否有可用的CUDA设备
    if torch.cuda.is_available():
        print("CUDA is available!")  # 输出CUDA是否可用
        device_count = torch.cuda.device_count()  # 获取可用的GPU数量
        print(f"Number of available GPUs: {device_count}")  # 输出可用的GPU数量

        # 遍历所有可用的GPU设备
        for i in range(device_count):
            print(f"\nDetails for GPU {i}:")  # 输出当前GPU的序号
            device_properties = torch.cuda.get_device_properties(i)  # 获取当前GPU的属性
            print(f"  Name: {device_properties.name}")  # 输出GPU名称
            print(f"  Total memory: {device_properties.total_memory / (1024 ** 3):.2f} GB")  # 输出GPU总内存
            print(f"  CUDA capability: {device_properties.major}.{device_properties.minor}")  # 输出CUDA计算能力
            print(f"  Multi-processor count: {device_properties.multi_processor_count}")  # 输出多处理器数量
            # 以下几行被注释掉了，可以根据需要取消注释以获取更多详细信息
            # print(f"  Clock rate: {device_properties.clock_rate / 1000:.2f} MHz")  # 输出时钟频率
            # print(f"  L2 cache size: {device_properties.l2_cache_size} bytes")  # 输出L2缓存大小
            # print(f"  Max threads per block: {device_properties.max_threads_per_block}")  # 输出每个块的最大线程数
            # print(f"  Max thread dimensions: {device_properties.max_threads_dim}")  # 输出最大线程维度
            # print(f"  Max grid dimensions: {device_properties.max_grid_dim}")  # 输出最大网格维度

        # 检查cuDNN是否可用
        if cudnn.is_available():
            print("\ncuDNN is available!")  # 输出cuDNN是否可用
            print(f"  cuDNN version: {cudnn.version()}")  # 输出cuDNN版本
        else:
            print("\ncuDNN is not available.")  # 输出cuDNN不可用
    else:
        print("CUDA is not available. Running on CPU.")  # 输出CUDA不可用，将在CPU上运行


# 如果脚本作为主程序运行，则调用check_gpu函数
if __name__ == "__main__":
    check_gpu()  # 调用check_gpu函数
