import torch
import deepspeed
import deepspeed.comm as dist
import time

deepspeed.init_distributed()

def alloc_tensors(use_dtype):
    a = torch.ones(1024, 1024, dtype=torch.bfloat16)
    c = torch.ones(1024, 1024, dtype=torch.bfloat16)
    #t = torch.ones(6144, 4, dtype=use_dtype) * (dist.get_rank()+1.0)
    #t = torch.ones(14336, 4, dtype=use_dtype) * (dist.get_rank()+1.0)
    #t = torch.ones(1024, 10, dtype=use_dtype) * (dist.get_rank()+1.0)
    t = torch.ones(1024, 5, dtype=use_dtype) * (dist.get_rank()+1.0)
    #t = torch.ones(1024, 512, dtype=use_dtype) * (dist.get_rank()+1.0)
    return a, c, t

def test_allreduce(reuse_buffer, use_dtype, loop_count):
    b = torch.ones(1024, 1024, dtype=torch.bfloat16)
    if reuse_buffer:
        a,c,t = alloc_tensors(use_dtype)
    t_total = 0.0
    start = time.time()
    for i in range(loop_count):
        if not reuse_buffer:
            a,c,t = alloc_tensors(use_dtype)
        #torch.matmul(a, b, out=c)
        #torch.matmul(a, b, out=c)
        #v = 0.0000001
        #for i in range(100000):
        #    v = v + 0.01
        dist.barrier(t)
        t0 = time.time()
        dist.inference_all_reduce(t, async_op=False)
        #dist.all_reduce(t)
        t1 = time.time()
        if (i==0 and dist.get_rank()==0):
            print (t)
        t_total += t1-t0
    end = time.time()
    #print (f'total elapsed {end-start}')
    return t_total

loop_count = 579200
t0 = time.time()
t = test_allreduce(False, torch.bfloat16, loop_count)
t1 = time.time()
#t = test_allreduce(False, torch.float32, loop_count)
#t = test_allreduce(False, torch.float16, loop_count)

if dist.get_rank() == 0:
    print (f'avg duration = {t/(loop_count/1000.0)}ms, E2E={t1-t0}')

#dist.log_summary(show_straggler=True)
