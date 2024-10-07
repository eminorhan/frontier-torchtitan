import os
import torch
from mpi4py import MPI
from datetime import timedelta

num_gpus_per_node = torch.cuda.device_count()
print ("num_gpus_per_node = " + str(num_gpus_per_node), flush=True)

comm = MPI.COMM_WORLD
world_size = comm.Get_size()
global_rank = rank = comm.Get_rank()
local_rank = int(rank) % int(num_gpus_per_node) # local_rank and device are 0 when using 1 GPU per task
backend = None
os.environ['WORLD_SIZE'] = str(world_size)
os.environ['RANK'] = str(global_rank)
os.environ['LOCAL_RANK'] = str(local_rank)
os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0'

torch.distributed.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size, timeout=timedelta(seconds=300))

print("Successfully initialized distributed training.")
