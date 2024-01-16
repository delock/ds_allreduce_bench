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

def alloc_tensors(num_bytes, use_dtype):
    # Calculate the number of elements that fit into num_bytes for the given data type
    element_size = torch.tensor([], dtype=use_dtype).element_size()
    num_elements = num_bytes // element_size

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

def test_allreduce_bak(reuse_buffer, use_dtype, loop_count):
    ref = result_tensor(use_dtype)
    b = torch.ones(1024, 1024, dtype=torch.bfloat16)
    a_ref,c_ref,t_ref = alloc_tensors(use_dtype)
    if reuse_buffer:
        a = a_ref.clone()
        c = c_ref.clone()
        t = t_ref.clone()
    t_total = 0.0
    rank = dist.get_rank()
    start = time.time()
    for i in range(loop_count):
        if not reuse_buffer:
            a = a_ref.clone()
            c = c_ref.clone()
            t = t_ref.clone()
        torch.matmul(a, b, out=c)
        dist.barrier()
        t0 = time.time()
        if os.environ.get('USE_ONECCL') == '1' or args.ccl:
            dist.all_reduce(t)
        else:
            dist.inference_all_reduce(t, async_op=False)
        t1 = time.time()
        if i==0:
            if rank == 0:
                print (f'[{rank}] max rel diff with ref {((ref-t)/ref).abs().max()}')
                print (t)
                root_result = t
            else:
                root_result = torch.empty(args.elements, dtype=use_dtype)
            dist.broadcast(root_result, 0)
            if (t-root_result).max() != torch.zeros(1, dtype=use_dtype):
                print (f'[{rank}] result diff with rank 0, correct allreduce must ensure identical result among all ranks')
        t_total += t1-t0
        if rank == 0:
            print (f'iteration {i} of {loop_count}', end='\r')
    end = time.time()
    return t_total

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
    # Convert local timings to tensor and gather all timings
    #local_timings_tensor = torch.tensor(local_timings, dtype=torch.float64)
    #all_timings_tensor = [torch.zeros_like(local_timings_tensor) for _ in range(world_size)]
    #dist.all_gather(all_timings_tensor, local_timings_tensor)

    # Convert list of tensors to a single tensor
    #all_timings_tensor = torch.stack(all_timings_tensor).view(-1)

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

    #print(f"{num_bytes:14d}{num_iterations:14d}{t_min:14.2f}{t_max:14.2f}{t_avg:14.2f}{stddev:12.2f}")
    
def create_csv_header():
    with open('output.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["#ranks", "collective", "reduction", "dtype", "dtype_size", "#elements/buffer",
                             "message_size", "#buffers", "#repetitions", "t_min[usec]", "t_max[usec]",
                             "t_avg[usec]", "stddev[%]", "wait_t_avg[usec]"])

def print_timings_csv(ranks, collective, reduction, dtype, dtype_size, num_elements_buffer,
                      message_size, num_buffers, num_repetitions, t_min, t_max, t_avg, stddev, wait_t_avg):
    with open('output.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([ranks, collective, reduction, dtype, dtype_size, num_elements_buffer,
                             message_size, num_buffers, num_repetitions, t_min, t_max, t_avg, stddev, wait_t_avg])


def test_allreduce(reuse_buffer, use_dtype, num_bytes_list, num_iterations, warmup_iters):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print_head(world_size)
    if rank == 0:
        # Create CSV header only once
        create_csv_header()

    for num_bytes in num_bytes_list:
        #print(f"Run for {num_bytes:14d} bytes and {num_iterations:14d} iterations")
        # Allocate tensors of the specified size
        #tensor_size = num_bytes // torch.tensor([], dtype=use_dtype).element_size()
        b = torch.ones(1024, 1024, dtype=torch.bfloat16)
        a_ref,c_ref,t_ref = alloc_tensors(num_bytes, use_dtype)
        if reuse_buffer:
            a = a_ref.clone()
            c = c_ref.clone()
            t = t_ref.clone()

        time_list = []  # List to store time values per iteration
        for _ in range(num_iterations + warmup_iters):
            if not reuse_buffer:
                a = a_ref.clone()
                c = c_ref.clone()
                t = t_ref.clone()

            #torch.matmul(a, b, out=c)
            dist.barrier()
            t0 = time.time()
            if os.environ.get('USE_ONECCL') == '1' or args.ccl:
                dist.all_reduce(t)
            else:
                dist.inference_all_reduce(t, async_op=False)
            t1 = time.time()

            if _ >= warmup_iters:
                time_list.append((t1 - t0) * 1e6)  # Convert to microseconds
            '''elif _ == 0:
                if rank == 0:
                    ref = result_tensor(num_bytes, use_dtype)
                    print (f'[{rank}] max rel diff with ref {((ref-t)/ref).abs().max()}')
                    print (t)
                    root_result = t
                else:
                    root_result = torch.empty(args.elements, dtype=use_dtype)
                dist.broadcast(root_result, 0)
                if (t-root_result).max() != torch.zeros(1, dtype=use_dtype):
                    print (f'[{rank}] result diff with rank 0, correct allreduce must ensure identical result among all ranks')
            '''
        # evaluate and print
        if rank == 0:
            print_timings(time_list, num_bytes, use_dtype, num_iterations)
        if rank == 0:
            # Calculate and print min, max, mean, and standard deviation
            t_min = np.min(time_list)
            t_max = np.max(time_list)
            t_avg = np.mean(time_list)
            stddev = np.std(time_list) / t_avg * 100  # Percentage
            num_buffers=1
            wait_t_avg=0
            element_size = torch.tensor([], dtype=use_dtype).element_size()
            num_elements = num_bytes // element_size
            print_timings_csv(world_size, "allreduce", "sum", use_dtype, element_size,
                              num_bytes, num_elements, num_buffers, num_repetitions, t_min, t_max, t_avg, stddev, wait_t_avg)


    return time_list
    
loop_count = args.count
warmup_iters = args.nwarmup
use_cache = args.cache
#t0 = time.time()
#t, tlist = test_allreduce_bak(use_cache, dtype, loop_count, warmup_iters)
#t1 = time.time()

#if dist.get_rank() == 0:
#    print_time_by_rank(tlist)
#    print (f'num_elements = {args.elements}, dtype = {dtype} allreduce use {"ccl" if args.ccl else "shm"}\navg duration = {t/(loop_count/1000.0)}ms')
#    min_time, max_time = find_min_max_times(tlist)

# Example usage:
num_repetitions = 1000
min_bytes = 4
max_bytes = nelem_to_bytes(args.elements, dtype)  # Not a power of two
num_bytes_list = generate_num_bytes_list(min_bytes, max_bytes)
print(num_bytes_list)

num_repetitions = 1000
test_allreduce(use_cache, dtype, num_bytes_list, num_repetitions, warmup_iters)
