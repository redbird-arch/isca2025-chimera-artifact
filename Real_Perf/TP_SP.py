import torch
import torch.distributed as dist

import argparse
import os
import time

dp = 1
sp = 8
pp = 1
tp = 8

embedding_size = 2048
sequence = 8192
# total seq = 8192 (not 8192/sp this test)
batch_size = 128
# total bs = 256
node_num = 8



def Original(input_tensor, rank):
    # output_tensor_list = [torch.empty_like(input_tensor) for _ in range(sp)]

    # dist.all_gather(output_tensor_list, input_tensor)
    # output_tensor_1 = torch.cat(output_tensor_list,dim=0)
    # del output_tensor_list
    # dist.barrier()
    dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)

    # dist.all_gather_into_tensor --- Gather tensors from all ranks and put them in a single output tensor.

    return 



def Fusion(input_tensor, rank):
    # output_tensor_list = [torch.empty_like(input_tensor) for _ in range(sp)]

    # dist.all_gather(output_tensor_list, input_tensor)
    # output_tensor_1 = torch.cat(output_tensor_list,dim=0)
    # del output_tensor_list

    # dist.barrier()
    # output_tensor_2 = torch.empty_like(input_tensor)

    # dist.all_reduce(output_tensor_1, op=ReduceOp.SUM)
    # dist.reduce_scatter_tensor(output_tensor_2, output_tensor_1)
    output_tensor = torch.zeros(int(input_tensor.shape[0]/tp), input_tensor.shape[1]).cuda()

    dist.barrier()
    start = time.time()
    torch.distributed._reduce_scatter_base(output_tensor, input_tensor)
    dist.barrier()
    if dist.get_rank()== 0:
        print('Fusion Time:', time.time()-start)


    

    # if "reduce_scatter_tensor" not in dir(torch.distributed):
    #     torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base
    # if "all_gather_into_tensor" not in dir(torch.distributed):
    #     torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base


    return 



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

    dist.barrier()
    start = time.time()
    output = Original(input_tensor, rank)
    dist.barrier()
    if dist.get_rank()== 0:
        print(f'Original Time:{time.time()-start}')
    

    # dist.barrier()
    # start = time.time()
    output = Fusion(input_tensor, rank)
    # dist.barrier()
    # if dist.get_rank()== 0:
    #     print('Fusion Time:', time.time()-start)


