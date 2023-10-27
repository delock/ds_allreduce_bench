import os
import torch
import deepspeed
import deepspeed.comm as dist
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--elements", type=int, default=16*1024)
parser.add_argument("--dtype", type=str, choices=["bf16", "fp32"], default="bf16")
parser.add_argument("--count", type=int, default=10000)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--ccl", action='store_true')
args = parser.parse_args()

if args.dtype=="bf16":
    dtype = torch.bfloat16
elif args.dtype=="fp32":
    dtype = torch.float32

deepspeed.init_distributed()

if dist.get_rank() == 0:
    for env in os.environ:
        print (f"'{env}': '{os.environ[env]}'")

def alloc_tensors(use_dtype):
    a = torch.ones(1024, 1024, dtype=torch.bfloat16)
    c = torch.ones(1024, 1024, dtype=torch.bfloat16)
    t = torch.ones(args.elements, dtype=use_dtype) * (dist.get_rank()+1.0) + torch.tensor([i/64.0 for i in range(args.elements)], dtype=use_dtype)
    return a, c, t

def result_tensor(use_dtype):
    result = torch.ones(args.elements, dtype=use_dtype) * ((dist.get_world_size()+1)*dist.get_world_size()/2) + torch.tensor([i/64.0 for i in range(args.elements)], dtype=use_dtype) * dist.get_world_size()
    return result

def test_allreduce(reuse_buffer, use_dtype, loop_count):
    ref = result_tensor(use_dtype)
    b = torch.ones(1024, 1024, dtype=torch.bfloat16)
    if reuse_buffer:
        a,c,t = alloc_tensors(use_dtype)
    t_total = 0.0
    start = time.time()
    for i in range(loop_count):
        if not reuse_buffer:
            a,c,t = alloc_tensors(use_dtype)
        torch.matmul(a, b, out=c)
        dist.barrier(t)
        t0 = time.time()
        if os.environ.get('USE_ONECCL') == '1' or args.ccl:
            dist.all_reduce(t)
        else:
            dist.inference_all_reduce(t, async_op=False)
        t1 = time.time()
        #if (i==0 and dist.get_rank()==0):
        if (i==0):
            if (dist.get_rank() == 0):
                print (f'[{dist.get_rank()}] max rel diff with ref {((ref-t)/ref).abs().max()}')
                print (t)
                root_result = t
            else:
                root_result = torch.empty(args.elements, dtype=use_dtype)
            dist.broadcast(root_result, 0)
            if (t-root_result).max() != torch.zeros(1, dtype=use_dtype):
                print (f'[{dist.get_rank()}] result diff with rank 0, correct allreduce must ensure identical result among all ranks')
        t_total += t1-t0
    end = time.time()
    return t_total

loop_count = args.count
t0 = time.time()
t = test_allreduce(False, dtype, loop_count)
t1 = time.time()

if dist.get_rank() == 0:
    print (f'num_elements = {args.elements}, dtype = {dtype} allreduce use {"ccl" if args.ccl else "shm"}\navg duration = {t/(loop_count/1000.0)}ms')
