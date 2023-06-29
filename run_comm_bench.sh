#!/bin/bash
#export KMP_BLOCKTIME=1
#export KMP_SETTINGS=1
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

#export CCL_ALLREDUCE=recursive_doubling
export CCL_PROCESS_LAUNCHER=none

export FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
#export CCL_ITT_LEVEL=1
export CCL_WORKER_COUNT=1

#export CCL_WORKER_AFFINITY=10,22,34,46,58,70,82,94
#deepspeed --force_multi --hostfile hostfile.txt --launcher=impi --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 run_generation_with_deepspeed_profile.py --device cpu --dtype bfloat16 --num-iter 10 --num-warmup 4 -m facebook/opt-350m --input-tokens 22 --max-new-tokens 100 --benchmark --token-latency --greedy                             
#8 ranks
#PSM3_DEVICES=self,nic

export CCL_WORKER_AFFINITY=10,22,34,46,58,70,82,94
deepspeed --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 ds_comm_bench.py

#export CCL_WORKER_AFFINITY=10,58
#deepspeed --num_accelerator 2 --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 ds_comm_bench.py
#8 ranks mpi launcher
#deepspeed --force_multi --hostfile=hostfile.txt --launcher impi --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 ds_comm_bench.py
#deepspeed --force_multi --hostfile=hostfile.txt --launcher impi --bind_cores_to_rank ds_comm_bench.py

#mpirun -ppn 8 ~/oneCCL/build/_install/examples/benchmark/benchmark -i 17920 -y 5120 -d bfloat16
