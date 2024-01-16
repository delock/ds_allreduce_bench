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
#deepspeed --bind_cores_to_rank ds_bench_modified.py $*
#deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype fp32 --elements 4194304 --ccl --cache
deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_bench_modified.py --dtype fp32 --elements 4194304 --cache
# set bind core list to bind to cores other than CCL_WORKER_AFFINITY
#deepspeed --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 ds_comm_bench.py $*

#multi node
#deepspeed --hostfile hostfile.txt --force_multi --launcher impi --bind_cores_to_rank ds_comm_bench.py $*

#oneccl benchmark
#mpirun -n 8 ~/oneCCL/build/examples/benchmark/benchmark -d bfloat16 -f 10240 -t 10240 -i 50

