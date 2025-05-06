import torch
import torch.distributed as dist

import argparse
import os
import time

dp = 2
sp = 1
pp = 1
tp = 4

embedding_size = 2048
sequence = 8192
# total seq = 8192
batch_size = 128
# total bs = 256
node_num = 8


def Original(input_tensor, rank, output_tensor1, output_tensor2):

    # Ensure that the group creation (torch.distributed.new_group) is called consistently across all processes. 
    # group creation cannot be in "if rank < int(node_num/2):"
    SP_group_1 = torch.distributed.new_group(ranks=[0,1,2,3])
    SP_group_2 = torch.distributed.new_group(ranks=[4,5,6,7])

    if rank < int(node_num/2):
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM, group=SP_group_1)
    else:
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM, group=SP_group_2)
    # dist.all_gather_into_tensor --- Gather tensors from all ranks and put them in a single output tensor.

    dist.all_to_all_single(output_tensor_2, output_tensor_1)

    return output_tensor_2

def Fusion(input_tensor, rank, output_tensor1, output_tensor2):

    # Ensure that the group creation (torch.distributed.new_group) is called consistently across all processes. 
    # group creation cannot be in "if rank < int(node_num/2):"
    SP_group_1 = torch.distributed.new_group(ranks=[0,1,2,3])
    SP_group_2 = torch.distributed.new_group(ranks=[4,5,6,7])

    if rank < int(node_num/2):
        torch.distributed._reduce_scatter_base(output_tensor_1, input_tensor, group=SP_group_1)
    else:
        torch.distributed._reduce_scatter_base(output_tensor_2, input_tensor, group=SP_group_2)
    # dist.all_gather_into_tensor --- Gather tensors from all ranks and put them in a single output tensor.

    dist.all_to_all_single(output_tensor_2, output_tensor_1)

    return output_tensor_2



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Parser')
    parser.add_argument("--local-rank", type=int, required=True, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    local_rank = args.local_rank
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    

    input_tensor = torch.rand(batch_size * sequence, embedding_size).cuda()
    output_tensor_1 = torch.zeros(int(input_tensor.shape[0]/tp), input_tensor.shape[1]).cuda()
    output_tensor_2 = torch.zeros(int(input_tensor.shape[0]/tp), input_tensor.shape[1]).cuda()

    dist.barrier()
    start = time.time()
    output = Original(input_tensor, rank, output_tensor_1, output_tensor_2)
    dist.barrier()
    if dist.get_rank()== 0:
        print(f'Original Time:{time.time()-start}')
    

    dist.barrier()
    start = time.time()
    output = Fusion(input_tensor, rank, output_tensor_1, output_tensor_2)
    dist.barrier()
    if dist.get_rank()== 0:
        print('Fusion Time:', time.time()-start)