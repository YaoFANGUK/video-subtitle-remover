import os
import json
import argparse
from shutil import copyfile
import torch
import torch.multiprocessing as mp

from backend.tools.train.trainer_sttn import Trainer
from backend.tools.train.utils_sttn import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-c', '--config', default='configs_sttn/youtube-vos.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
args = parser.parse_args()


def main_worker(rank, config):
    # 如果配置中没有提到局部排序（local_rank），就给它和全局排序（global_rank）赋值为传入的排序（rank）
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank

    # 如果配置指定为分布式训练
    if config['distributed']:
        # 设置使用的CUDA设备为当前的本地排名对应的GPU
        torch.cuda.set_device(int(config['local_rank']))
        # 初始化分布式进程组，通过nccl后端
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=config['init_method'],
            world_size=config['world_size'],
            rank=config['global_rank'],
            group_name='mtorch'
        )
        # 打印当前GPU的使用情况，输出全球排名和本地排名
        print('using GPU {}-{} for training'.format(
            int(config['global_rank']), int(config['local_rank']))
        )

    # 创建模型保存的目录路径，包括模型名和配置文件名
    config['save_dir'] = os.path.join(
        config['save_dir'], '{}_{}'.format(config['model'], os.path.basename(args.config).split('.')[0])
    )

    # 如果CUDA可用，则设置设备为相应的CUDA设备，否则为CPU
    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'

    # 如果不是分布式训练，或者是分布式训练的主节点（rank 0）
    if (not config['distributed']) or config['global_rank'] == 0:
        # 创建模型保存目录，并允许如果该目录存在则忽略创建（exist_ok=True）
        os.makedirs(config['save_dir'], exist_ok=True)
        # 设置配置文件的保存路径
        config_path = os.path.join(
            config['save_dir'], config['config'].split('/')[-1]
        )
        # 如果配置文件不存在，则从给定的配置文件路径复制到新路径
        if not os.path.isfile(config_path):
            copyfile(config['config'], config_path)
        # 打印创建目录的信息
        print('[**] create folder {}'.format(config['save_dir']))

    # 初始化训练器，传入配置参数和debug标记
    trainer = Trainer(config, debug=args.exam)
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    # 加载配置文件
    config = json.load(open(args.config))
    config['model'] = args.model  # 设置模型名称
    config['config'] = args.config  # 设置配置文件路径

    # 设置分布式训练的相关配置
    config['world_size'] = get_world_size()  # 获取全局进程数，即训练过程中参与计算的总GPU数量
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"  # 设置初始化方法，包括主节点IP和端口
    config['distributed'] = True if config['world_size'] > 1 else False  # 根据世界规模确定是否启用分布式训练

    # 设置分布式并行训练环境
    if get_master_ip() == "127.0.0.1":
        # 如果主节点IP是本机地址，那么手动启动多个分布式训练进程
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config,))
    else:
        # 如果是由其他工具如OpenMPI启动的多个进程，不需手动创建进程。
        config['local_rank'] = get_local_rank()  # 获取本地（单个节点）排名
        config['global_rank'] = get_global_rank()  # 获取全局排名
        main_worker(-1, config)  # 启动主工作函数
