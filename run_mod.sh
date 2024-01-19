#!/bin/bash
#export KMP_BLOCKTIME=1
#export KMP_SETTINGS=1
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

#export CCL_ALLREDUCE=recursive_doubling
export CCL_PROCESS_LAUNCHER=none

export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
#export CCL_ITT_LEVEL=1
export CCL_WORKER_COUNT=1

#if turn this line on, need to use a small iteration count such as 50
#export CCL_SCHED_PROFILE=1

#for 48 core *2
#set CCL_WORKER_AFFINITY if necessary
#export CCL_WORKER_AFFINITY=10,22,34,46,58,70,82,94

#single node
#inference_all_reduce without cache
    echo "    default launcher, with compute, without cache"
    deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072 --compute
    deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 32768 --compute

    echo "    default launcher, without compute, without cache"
    deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072
    deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 32768

    echo "    impi launcher, with compute, without cache"
    deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072 --compute
    deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 32768 --compute

    echo "    impi launcher, without compute, without cache"
    deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072
    deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 32768

#inference_all_reduce with cache
    echo "    default launcher, with compute, with cache"
    deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072 --compute --cache
    deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 32768 --compute --cache

    echo "    default launcher, without compute, with cache"
    deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072 --cache
    deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 32768 --cache

    echo "    impi launcher, with compute, with cache"
    deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072 --compute --cache
    deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 32768 --compute --cache

    echo "    impi launcher, without compute, with cache"
    deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072 --cache
    deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 32768 --cache

#oneccl benchmark
#mpirun -n 8 ~/oneCCL/build/examples/benchmark/benchmark -d bfloat16 -f 10240 -t 10240 -i 50

