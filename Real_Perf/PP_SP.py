import torch
import torch.distributed as dist

import argparse
import os
import time

dp = 1
sp = 4
pp = 2

embedding_size = 2048
sequence = 2048
# total seq = 8192
batch_size = 128
# total bs = 256
node_num = 8

PP_Group_1 = [0,2,4,6]
PP_Group_2 = [1,3,5,7]


def Original(input_tensor, rank):
    output_tensor_list = [torch.empty_like(input_tensor) for _ in range(sp)]

    # Ensure that the group creation (torch.distributed.new_group) is called consistently across all processes. 
    # group creation cannot be in "if rank < int(node_num/2):"
    SP_group_1 = torch.distributed.new_group(ranks=PP_Group_1)
    # SP_group_2 = torch.distributed.new_group(ranks=[4,5,6,7])

    if rank in PP_Group_1:
        dist.all_gather(output_tensor_list, input_tensor, group=SP_group_1)
    # else:
    #     dist.all_gather(output_tensor_list, input_tensor, group=SP_group_2)

    # dist.all_gather_into_tensor --- Gather tensors from all ranks and put them in a single output tensor.

    output_tensor_1 = torch.cat(output_tensor_list,dim=0)
    # del output_tensor_list

    if rank in PP_Group_1:
        dist.send(tensor=output_tensor_1, dst=PP_Group_2[PP_Group_1.index(rank)])
    else:
        dist.recv(tensor=output_tensor_1, src=PP_Group_1[PP_Group_2.index(rank)])
    
    return output_tensor_1



def Fusion(input_tensor, rank):
    output_tensor = [torch.empty_like(input_tensor) for _ in range(sp)]

    reqs = []

    if rank in PP_Group_1:
        for i in range(int(node_num/2)):
            req = dist.isend(tensor=input_tensor, dst=PP_Group_2[i])
            reqs.append(req)
    else:
        for i in range(int(node_num/2)):
            req = dist.irecv(tensor=output_tensor[i], src=PP_Group_1[i])
            reqs.append(req)

    for req in reqs:
        req.wait()

    return output_tensor



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


