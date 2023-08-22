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

#for 48 core *2
export CCL_WORKER_AFFINITY=10,22,34,46,58,70,82,94
deepspeed --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 ds_comm_bench.py
#for 56 core *2
#export CCL_WORKER_AFFINITY=12,26,40,54,68,82,96,110
#deepspeed --bind_cores_to_rank --bind_core_list 0-11,14-25,28-39,42-53,56-67,70-81,84-95,98-109 ds_comm_bench.py
#mpirun -ppn 8 ~/oneCCL/build/_install/examples/benchmark/benchmark -i 17920 -y 24576 -d bfloat16

#export CCL_WORKER_AFFINITY=48,104
#deepspeed --num_accelerators 2 --bind_cores_to_rank --bind_core_list 0-39,48-87 ds_comm_bench.py
#deepspeed --num_accelerators 2 --bind_cores_to_rank --bind_core_list 0-47,56-103 ds_comm_bench.py
#mpirun -ppn 2 ~/oneCCL/build/_install/examples/benchmark/benchmark -i 17920 -y 57344 -d bfloat16

#deepspeed --num_accelerators 10 --bind_cores_to_rank ds_comm_bench.py

#export CCL_WORKER_AFFINITY=10,58
#deepspeed --num_accelerator 2 --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 ds_comm_bench.py
#8 ranks mpi launcher
#deepspeed --force_multi --hostfile=hostfile.txt --launcher impi --bind_cores_to_rank --bind_core_list 0-9,12-21,24-33,36-45,48-57,60-69,72-81,84-93 ds_comm_bench.py
#deepspeed --force_multi --hostfile=hostfile.txt --launcher impi --bind_cores_to_rank ds_comm_bench.py

