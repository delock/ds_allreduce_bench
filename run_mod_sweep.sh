#!/bin/bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib

deepspeed --bind_cores_to_rank ds_bench_modified.py --dtype bf16 --elements 16515072 --compute --elementlist $*

