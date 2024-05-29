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
# default core binding
#deepspeed --bind_cores_to_rank ds_comm_bench.py $*
deepspeed --num_accelerators 2 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 $* 2>&1 |tee result.txt
deepspeed --num_accelerators 2 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 3 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 3 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 6 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 6 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 11 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 11 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 16 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 16 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 17 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 17 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 2 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 2 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 3 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 3 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 6 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 6 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 11 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 11 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 16 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 16 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 17 --bind_cores_to_rank ds_comm_bench.py --elements 1024 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
deepspeed --num_accelerators 17 --bind_cores_to_rank ds_comm_bench.py --elements 1048576 --count 100 --dtype fp32 $* 2>&1 |tee -a result.txt
cat result.txt|grep diff
