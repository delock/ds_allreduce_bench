import os
import sys
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
import time
import argparse
try:
    import intel_extension_for_pytorch as ipex
    ipex_loaded = True
except ImportError:
    ipex_loaded = False

parser = argparse.ArgumentParser()
parser.add_argument("--elements", type=int, default=16*1024)
parser.add_argument("--dtype", type=str, choices=["bf16", "fp32", "fp16"], default="bf16")
parser.add_argument("--count", type=int, default=10000)
parser.add_argument("--warmup", type=int, default=1000)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--ccl", action='store_true')
parser.add_argument("--ipex", action='store_true')
args = parser.parse_args()

if args.dtype=="bf16":
    dtype = torch.bfloat16
elif args.dtype=="fp32":
    dtype = torch.float32
elif args.dtype=="fp16":
    dtype = torch.float16

deepspeed.init_distributed()
torch.manual_seed(dist.get_rank())

if dist.get_rank() == 0:
    for env in os.environ:
        print (f"'{env}': '{os.environ[env]}'")

def alloc_tensors(use_dtype):
    a = torch.ones(1024, 1024, dtype=torch.bfloat16).to(get_accelerator().device_name(dist.get_rank()))
    c = torch.ones(1024, 1024, dtype=torch.bfloat16).to(get_accelerator().device_name(dist.get_rank()))
    t_fp = torch.rand(args.elements, dtype=torch.float).to(get_accelerator().device_name(dist.get_rank()))
    t = t_fp.to(dtype=use_dtype).clone().to(get_accelerator().device_name(dist.get_rank()))
    return a, c, t, t_fp

def test_allreduce(reuse_buffer, use_dtype, loop_count):
    b = torch.ones(1024, 1024, dtype=torch.bfloat16).to(get_accelerator().device_name(dist.get_rank()))
    a_ref,c_ref,t_ref,t_fp_ref = alloc_tensors(use_dtype)
    if reuse_buffer:
        a = a_ref.clone()
        c = c_ref.clone()
        t = t_ref.clone()
    t_total = 0.0
    rank = dist.get_rank()
    for i in range(loop_count+args.warmup):
        if i==args.warmup:
            start = time.time()
        if not reuse_buffer:
            a = a_ref.clone()
            c = c_ref.clone()
            t = t_ref.clone()
        torch.matmul(a, b, out=c)
        dist.barrier()
        t0 = time.time()
        if os.environ.get('USE_ONECCL') == '1' or args.ccl:
            dist.all_reduce(t)
        elif args.ipex:
            if ipex_loaded:
                ipex.llm.distributed.all_reduce_add(t)
            else:
                print ("Intel Extension for PyTorch is not installed yet.");
                sys.exit(1)
        else:
            dist.inference_all_reduce(t)
        t1 = time.time()
        if i==0:
            first_time_result = t.clone()
            dist.all_reduce(t_fp_ref)
            if rank == 0:
                print (f'[{rank}] max rel diff with ref {((t_fp_ref-t)/t_fp_ref).abs().max()}')
                print (t)
                root_result = t
            else:
                root_result = torch.empty(args.elements, dtype=use_dtype, device=get_accelerator().device_name(rank))
            dist.broadcast(root_result, 0)
            if (t-root_result).abs().max() != torch.zeros(1, dtype=use_dtype, device=get_accelerator().device_name(rank)):
                print (f'[{rank}] result diff with rank 0, correct allreduce must ensure identical result among all ranks')
        if (t-first_time_result).abs().max() != torch.zeros(1, dtype=use_dtype, device=get_accelerator().device_name(rank)):
                print (f'[{rank}] result diff with first time result, correct allreduce must ensure identical result between runs.  Max diff={(t-first_time_result).abs().max()}')
        t_total += t1-t0
        if rank == 0:
            print (f'iteration {i} of {loop_count}', end='\r')
    end = time.time()
    return t_total

loop_count = args.count
t0 = time.time()
t = test_allreduce(False, dtype, loop_count)
t1 = time.time()

if dist.get_rank() == 0:
    print (f'num_elements = {args.elements}, dtype = {dtype} allreduce use {"ccl" if args.ccl else "shm"}\navg duration = {t/(loop_count/1000.0)}ms')
