import torch
import torch.distributed as dist

import argparse
import os
import time

dp = 1
sp = 1
pp = 2
tp = 4

embedding_size = 2048
sequence = 8192
# total seq = 8192
batch_size = 64
# total bs = 256
node_num = 8



def Original(input_tensor, rank, multicast_list1, output_tensor_list1, output_tensor1):

    SP_group_1 = torch.distributed.new_group(ranks=[0,1,2,3])
    SP_group_2 = torch.distributed.new_group(ranks=[4,5,6,7])

    if rank < int(node_num/2):
        dist.all_reduce(input_tensor, op=dist.reduce_op.SUM, group=SP_group_1)

    reqs = []
    if rank < int(node_num/2):
        for i in range(int(node_num/2)):
            req = dist.isend(tensor=multicast_list1[tp*rank+i], dst= int(node_num/2) + i)
            reqs.append(req)
    else:
        for i in range(int(node_num/2)):
            req = dist.irecv(tensor=output_tensor_list1[i], src=i)
            reqs.append(req)

    for req in reqs:
        req.wait()

    # del multicast_list
    # del output_tensor_list

    all_gather_list = [torch.empty_like(output_tensor1) for _ in range(tp)]
    dist.all_gather(all_gather_list, output_tensor1, group=SP_group_2)

    return 


def Fusion(input_tensor, rank, output_tensor2, multicast_list2, output_tensor_list2, output_tensor_2):
    SP_group_1 = torch.distributed.new_group(ranks=[0,1,2,3])
    SP_group_2 = torch.distributed.new_group(ranks=[4,5,6,7])

    
    torch.distributed._reduce_scatter_base(output_tensor2, input_tensor, group=SP_group_1)
    reqs = []

    if rank < int(node_num/2):
        for i in range(int(node_num/2)):
            req = dist.isend(tensor=multicast_list2[i], dst= int(node_num/2) + i)
            reqs.append(req)
    else:
        for i in range(int(node_num/2)):
            req = dist.irecv(tensor=output_tensor_list2[i], src=i)
            reqs.append(req)

    for req in reqs:
        req.wait()

    all_gather_list = [torch.empty_like(output_tensor_2) for _ in range(tp)]
    dist.all_gather(all_gather_list, output_tensor_2, group=SP_group_2)

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
    
    multicast_list1 = list(input_tensor.chunk(tp*tp))
    output_tensor_list1 = [torch.zeros(int(input_tensor.shape[0]/(tp*tp)), input_tensor.shape[1]).cuda() for _ in range(tp)]
    output_tensor1 = torch.cat(output_tensor_list1,dim=0)

    output_tensor2 = torch.zeros(int(input_tensor.shape[0]/tp), input_tensor.shape[1]).cuda()
    multicast_list2 = list(output_tensor2.chunk(tp))
    output_tensor_list2 = [torch.empty_like(multicast_list2[0]) for _ in range(tp)]
    output_tensor_2 = torch.cat(output_tensor_list2,dim=0)



    dist.barrier()
    start = time.time()
    output = Original(input_tensor, rank, multicast_list1, output_tensor_list1, output_tensor1)
    dist.barrier()
    if dist.get_rank()== 0:
        print(f'Original Time:{time.time()-start}')
    

    dist.barrier()
    start = time.time()
    output = Fusion(input_tensor, rank, output_tensor2, multicast_list2, output_tensor_list2, output_tensor_2)
    dist.barrier()
    if dist.get_rank()== 0:
        print('Fusion Time:', time.time()-start)


