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



def Original(input_tensor, rank):
    
    SP_group_1 = torch.distributed.new_group(ranks=[0,1,2,3])
    SP_group_2 = torch.distributed.new_group(ranks=[4,5,6,7])

    # 2.866s
    if rank < int(node_num/2):
        dist.all_reduce(input_tensor, op=dist.reduce_op.SUM, group=SP_group_1)
    else:
        dist.all_reduce(input_tensor, op=dist.reduce_op.SUM, group=SP_group_2)

    # dist.barrier()

    SP_group_3 = torch.distributed.new_group(ranks=[0,4])
    SP_group_4 = torch.distributed.new_group(ranks=[1,5])
    SP_group_5 = torch.distributed.new_group(ranks=[2,6])
    SP_group_6 = torch.distributed.new_group(ranks=[3,7])

    # all2all_list = list(input_tensor.chunk(tp))

    output_tensor = torch.empty_like(input_tensor)

    # 1s
    if rank == 0 or rank == 4:
        dist.all_to_all_single(output_tensor, input_tensor, group=SP_group_3)
    elif rank == 1 or rank == 5:
        dist.all_to_all_single(output_tensor, input_tensor, group=SP_group_4)
    elif rank == 2 or rank == 6:
        dist.all_to_all_single(output_tensor, input_tensor, group=SP_group_5)
    elif rank == 3 or rank == 7:
        dist.all_to_all_single(output_tensor, input_tensor, group=SP_group_6)

    return 

# / need int(/)
def Fusion(input_tensor, rank):
    SP_group_1 = torch.distributed.new_group(ranks=[0,1,2,3])
    SP_group_2 = torch.distributed.new_group(ranks=[4,5,6,7])

    output_tensor = torch.zeros(int(input_tensor.shape[0]/tp), input_tensor.shape[1]).cuda()
    
    # 2s
    if rank < int(node_num/2):
        torch.distributed._reduce_scatter_base(output_tensor, input_tensor, group=SP_group_1)
    else:
        torch.distributed._reduce_scatter_base(output_tensor, input_tensor, group=SP_group_2)

    # dist.barrier()

    output_tensor_2 = torch.empty_like(output_tensor)
    # 1.8050880432128906s
    dist.all_to_all_single(output_tensor_2, output_tensor)

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
    

    dist.barrier()
    start = time.time()
    output = Fusion(input_tensor, rank)
    dist.barrier()
    if dist.get_rank()== 0:
        print('Fusion Time:', time.time()-start)


