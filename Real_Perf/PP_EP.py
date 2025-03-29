import torch
import torch.distributed as dist

import argparse
import os
import time



embedding_size = 2048
sequence = 8192
batch_size = 64

node_num = 8

def P2P_All2All(input_tensor, rank, group):
    output_tensor = torch.empty_like(input_tensor)

    # torch.cuda.synchronize()
    dist.barrier()
    start = time.time()

    if rank < int(node_num/2):
        dist.send(tensor=input_tensor, dst=rank+int(node_num/2))
    else:
        dist.recv(tensor=input_tensor, src=rank-int(node_num/2))
        dist.all_to_all_single(output_tensor, input_tensor, group=group)
    
    # torch.cuda.synchronize()
    dist.barrier()
    if dist.get_rank()== 0:
        print(f'P2P_All2All Time:{time.time()-start}')
    # print(f'Rank{dist.get_rank()}_P2P_All2All Time:{time.time()-start}')

    # print(f"TEST: RANK{rank}:{output_tensor}")
    return output_tensor

def Fused_Multicast(input_tensor, rank, group):
    '''
    baseline
    '''

    output_tensor = torch.empty_like(input_tensor)

    scatter_list = list(input_tensor.chunk(int(node_num/2)))
    gather_list  = list(output_tensor.chunk(int(node_num/2)))

    # for i in range(2):
    #     dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i, group=group)
    #     # RuntimeError: Global rank 0 is not part of group <torch.distributed.distributed_c10d.ProcessGroupNCCL object at 0x7f59e6539330>
    

    dist.barrier()
    # torch.cuda.synchronize()
    start = time.time()

    reqs = []

    if rank < int(node_num/2):
        for i in range(int(node_num/2)):
            req = dist.isend(tensor=scatter_list[i], dst= int(node_num/2) + i)
            reqs.append(req)
    else:
        for i in range(int(node_num/2)):
            req = dist.irecv(tensor=gather_list[i], src=i)
            reqs.append(req)

    for req in reqs:
        req.wait()

    # torch.cuda.synchronize()
    dist.barrier()
    if dist.get_rank()== 0:
        print('Fused_Multicast Time:', time.time()-start)
    
    # print(f'Rank{rank}_Fused_Multicast Time:{time.time()-start}')

    # print(f"END: RANK{rank}:{gather_list}")
    
    return gather_list


# def Fused_Multicast_opt_1(input_tensor, rank, group):
#     '''
#     reorder
#     '''
#     output_tensor = torch.empty_like(input_tensor)

#     scatter_list = list(input_tensor.chunk(int(node_num/2)))
#     gather_list  = list(output_tensor.chunk(int(node_num/2)))
    
#     dist.barrier()
#     # torch.cuda.synchronize()
#     start = time.time()

#     reqs = []

#     if rank < int(node_num/2):
#         for i in range(int(node_num/2)):
#             req = dist.isend(tensor=scatter_list[i], dst= int(node_num/2) + (rank+i)%int(node_num/2))
#             reqs.append(req)
#     else:
#         for i in range(int(node_num/2)):
#             req = dist.irecv(tensor=gather_list[i], src=i)
#             reqs.append(req)

#     for req in reqs:
#         req.wait()

#     # torch.cuda.synchronize()
#     dist.barrier()
#     if dist.get_rank()== 0:
#         print('Fused_Multicast Time:', time.time()-start)
#     # print(f'Rank{rank}_Fused_Multicast Time:{time.time()-start}')
#     return gather_list


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
    
    All2All_nodes = [i for i in range(int(node_num/2), int(node_num))]
    All2All_group = torch.distributed.new_group(ranks=All2All_nodes)

    input_tensor = torch.rand(batch_size * sequence, embedding_size).cuda()
    
    # print(f"START: RANK{rank}:{input_tensor}")
    output = P2P_All2All(input_tensor, rank, All2All_group)
    # print(f"END-P2P_All2All: RANK{rank}:{output}")

    output = Fused_Multicast(input_tensor, rank, All2All_group)
    # output = Fused_Multicast_opt_1(input_tensor, rank, All2All_group)

    # print(f"END-Fused_Multicast: RANK{rank}:{output}")



