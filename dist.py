import torch
import argparse
from torch import distributed as dist

print(torch.cuda.device_count())  # 打印gpu数量
torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
print('world_size', torch.distributed.get_world_size()) # 打印当前进程数

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int)  
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
