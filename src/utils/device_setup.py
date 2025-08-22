import multiprocessing
import os


def setup_device():
    print("Configuring for CPU")

    n_cores = multiprocessing.cpu_count()
    env_vars = {
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={n_cores} "
        f"inter_op_parallelism_threads={n_cores} "
        f"--xla_cpu_multi_thread_eigen=true "
        f"--xla_cpu_allocator=platform "
        "JAX_ENABLE_X64": "1",
        "OMP_NUM_THREADS": str(n_cores),
        "MKL_NUM_THREADS": str(n_cores),
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_TRACEBACK_FILTERING": "off",
    }

    # Set the environment variables
    for key, val in env_vars.items():
        os.environ[key] = val
