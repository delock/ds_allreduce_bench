# usage:
1. modify run_comm_bench.sh and modify parameters as you need, especially bind_core_list to fit your machine configuration
2. run with the following args:
    run_comm_bench.sh [args]
* --elements: number of elements, default 16384
* --dtype: data type (bf16 or fp32), default bf16
* --count: number of iterations, default 10000
* --ccl: if present use oneccl allreduce, otherwise call deepspeed inference_all_reduce which intend to optimize for inference
3. Alternative way of benchmarking is `run_mod_sweep.sh` which generates report that is more oneCCL flavor

# validate correctness:
* `./validate.sh`
* `./validate.sh --ccl` -- validate ccl implementation
* `./validate.sh --ipex` -- validate ipex implementation

# performance test:
* `./run_mod_sweep.sh` -- test inference_all_reduce performance, which is deepspeed built-in implmentation
* `./run_mod_sweep.sh --ipex` -- test ipex performance
For the following two, you need to have oneCCL binding for PyTorch and oneCCL installed in your environment
* `./run_mod_sweep.sh --ccl` -- test ccl performance
* `./run_mod_sweep.sh --torch` -- test torch.distributed performance, when oneCCL binding for PyTorch is installed, this will test allreduce performance through oneCCL binding for PyTorch.
