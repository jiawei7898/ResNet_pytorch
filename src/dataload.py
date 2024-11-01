import os
import urllib.request
import tarfile

# 定义数据集的下载链接
DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'


def download_data(url, dest_dir):
    """
    下载并解压数据集到指定目录。

    :param url: 数据集下载链接
    :param dest_dir: 目标目录
    """
    # 获取文件名
    filename = os.path.basename(url)
    filepath = os.path.join(dest_dir, filename)

    # 检查文件是否已经存在
    if os.path.exists(filepath):
        print(f"文件 {filename} 已经存在于 {filepath}，跳过下载。")
    else:
        # 下载文件
        print(f"开始下载 {filename} 到 {filepath}...")
        urllib.request.urlretrieve(url, filepath, reporthook=download_progress)
        print(f"{filename} 下载完成！")
    # 同样的检查是否解压
    if not os.path.exists(os.path.join(dest_dir, filename)):
        # 解压文件
        print("正在解压文件...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=dest_dir)
        print("解压完成！")
    else:
        print("文件已解压，跳过解压。")


def download_progress(count, block_size, total_size):
    """
    显示下载进度。

    :param count: 已经传输的数据块数目
    :param block_size: 数据块大小（一般为读取缓冲区大小）
    :param total_size: 远程文件大小
    """
    # 计算下载进度百分比
    percent = int(count * block_size * 100 / total_size)
    # 打印进度条
    print(f"\r{'=' * int(percent / 2)}>{'.' * (50 - int(percent / 2))} {percent}% ", end="")


# 主程序入口
if __name__ == '__main__':
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置数据集保存的目录
    data_dir = os.path.join(current_dir, '../data')  # 将数据集保存到项目的 data 目录

    # 创建数据目录（如果不存在）
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 调用函数下载数据集
    download_data(DATA_URL, data_dir)
