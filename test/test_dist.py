import os
import argparse
import torch
from mpi4py import MPI
from datetime import timedelta

parser = argparse.ArgumentParser(description='simple distributed training job')
parser.add_argument("--master_addr", type=str, required=True)
parser.add_argument("--master_port", type=str, required=True)

args = parser.parse_args()

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
os.environ['MASTER_ADDR'] = str(args.master_addr)
os.environ['MASTER_PORT'] = str(args.master_port)
torch.distributed.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size, timeout=timedelta(seconds=300))

print("Successfully initialized distributed training.")
