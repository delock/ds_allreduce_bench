import os
import torch
import deepspeed
import deepspeed.comm as dist
import time
import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--elements", type=int, default=16*1024)
parser.add_argument("--dtype", type=str, choices=["bf16", "fp32"], default="bf16")
parser.add_argument("--nwarmup", type=int, default=16)
parser.add_argument("--count", type=int, default=10000)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--ccl", action='store_true')
parser.add_argument("--compute", action='store_true')
parser.add_argument("--cache", action='store_true', default=False)
args = parser.parse_args()

if args.dtype=="bf16":
    dtype = torch.bfloat16
elif args.dtype=="fp32":
    dtype = torch.float32

deepspeed.init_distributed()

if dist.get_rank() == 0:
    for env in os.environ:
        print (f"'{env}': '{os.environ[env]}'")

def alloc_tensors(num_elements, use_dtype):
    # Allocate tensor 't' with the calculated number of elements
    t = torch.ones(num_elements, dtype=use_dtype) * (dist.get_rank() + 1.0)
    t += torch.tensor([i / 64.0 for i in range(num_elements)], dtype=use_dtype)

    a = torch.ones(1024, 1024, dtype=torch.bfloat16)
    c = torch.ones(1024, 1024, dtype=torch.bfloat16)
    #t = torch.ones(args.elements, dtype=use_dtype) * (dist.get_rank()+1.0) + torch.tensor([i/64.0 for i in range(args.elements)], dtype=use_dtype)
    return a, c, t

def result_tensor(num_bytes, use_dtype):
    element_size = torch.tensor([], dtype=use_dtype).element_size()
    num_elements = num_bytes // element_size
    result = torch.ones(num_elements, dtype=use_dtype) * ((dist.get_world_size()+1)*dist.get_world_size()/2) + torch.tensor([i/64.0 for i in range(num_elements)], dtype=use_dtype) * dist.get_world_size()
    return result

def nelem_to_bytes(num_elements, use_dtype):
    # Calculate the number of elements that fit into num_bytes for the given data type
    element_size = torch.tensor([], dtype=use_dtype).element_size()
    num_bytes = num_elements * element_size

    return num_bytes

def generate_num_bytes_list(min_bytes, max_bytes):
    num_bytes_list = []
    current_bytes = min_bytes
    while current_bytes < max_bytes:
        num_bytes_list.append(current_bytes)
        current_bytes *= 2
    num_bytes_list.append(max_bytes)  # Ensure max_bytes is included in the list
    return num_bytes_list

def find_min_max_times(time_list):
    # Extract just the time values from the list of (rank, time) pairs
    time_values = [time for rank, time in time_list]

    # Find the minimum and maximum time values
    min_time = min(time_values)
    max_time = max(time_values)
    print(f'Minimum time across all ranks: {min_time*1000.0:.6f} ms')
    print(f'Maximum time across all ranks: {max_time*1000.0:.6f} ms')

    return min_time, max_time

def print_time_by_rank(time_list):
    for rank, time_value in time_list:
        if rank > 0:
            print(f'Rank {rank}: Time {time_value*1000.0:.6f} ms')

# Function to print timings
def print_head(world_size):
    print("#------------------------------------------------------------")
    print("# Benchmarking: allreduce")
    print(f"# #processes: {world_size}")
    print("#------------------------------------------------------------")
    print()
    print("        #bytes  #repetitions   t_min[usec]   t_max[usec]   t_avg[usec]  stddev[%]")

# Function to print timings
def print_timings(local_timings, num_elements, dtype, num_iterations):
    # Compute min, max, average, and standard deviation
    # Calculate min, max, average, and standard deviation
    t_min = np.min(local_timings)
    t_max = np.max(local_timings)
    t_avg = np.mean(local_timings)
    stddev = np.std(local_timings) / t_avg * 100  # Percentage

    # Print results
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = num_elements * bytes_per_element
    print(f"{total_bytes:14d}{num_iterations:14d}{t_min:14.2f}{t_max:14.2f}{t_avg:14.2f}{stddev:12.2f}")

def test_allreduce(reuse_buffer, use_dtype, num_elms_list, num_iterations, warmup_iters):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print_head(world_size)

    for num_elms in num_elms_list:
        # Allocate tensors of the specified size
        b = torch.ones(1024, 1024, dtype=torch.bfloat16)
        a_ref,c_ref,t_ref = alloc_tensors(num_elms, use_dtype)
        if reuse_buffer:
            a = a_ref.clone()
            c = c_ref.clone()
            t = t_ref.clone()

        time_list = []  # List to store time values per iteration
        print (t_ref.size())
        print (t_ref.dtype)
        for _ in range(num_iterations + warmup_iters):
            if not reuse_buffer:
                a = a_ref.clone()
                c = c_ref.clone()
                t = t_ref.clone()

            if args.compute:
                torch.matmul(a, b, out=c)
            dist.barrier()
            t0 = time.time()
            if os.environ.get('USE_ONECCL') == '1' or args.ccl:
                dist.all_reduce(t)
            else:
                dist.inference_all_reduce(t, async_op=False)
            t1 = time.time()

            if _ >= warmup_iters:
                time_list.append((t1 - t0) * 1e6)  # Convert to microseconds

        # evaluate and print
        if rank == 0:
            print_timings(time_list, num_elms, use_dtype, num_iterations)

    return time_list

loop_count = args.count
warmup_iters = args.nwarmup
use_cache = args.cache

num_elms_list = [args.elements]
print(num_elms_list)

num_repetitions = 1000
test_allreduce(use_cache, dtype, num_elms_list, num_repetitions, warmup_iters)
