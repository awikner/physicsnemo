
import tracemalloc
import functools
import torch
import torch.distributed as dist


# dist.init_process_group("nccl")
# rank = dist.get_rank()
# world_size = dist.get_world_size()

def log_memory_usage(rank):
    def decorator(func):
        """
        A decorator to log the memory usage of the decorated function.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()  # Start tracing memory allocations

            result = func(*args, **kwargs)  # Execute the decorated function
            if rank == 0:
                snapshot = tracemalloc.take_snapshot()  # Take a snapshot of memory allocations
                tracemalloc.stop()  # Stop tracing memory allocations

                top_stats = snapshot.statistics('lineno') # Get statistics sorted by line number

                print(f"Memory usage for function '{func.__name__}':")
                for stat in top_stats[:5]:  # Print top 5 memory-consuming lines
                    print(f"  {stat}")
                return result
        return wrapper
    return decorator


def log_gpu_memory(func):
    """
    A decorator to log the GPU memory usage of the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial GPU memory usage
        torch.cuda.empty_cache()  # Clear cache to get a more accurate baseline
        initial_allocated = torch.cuda.memory_allocated() / (1024**2) # MB
        initial_cached = torch.cuda.memory_reserved() / (1024**2) # MB

        print(f"[{func.__name__}] Initial GPU memory allocated: {initial_allocated:.2f} MB, cached: {initial_cached:.2f} MB")

        # Execute the decorated function
        result = func(*args, **kwargs)

        # Get final GPU memory usage
        final_allocated = torch.cuda.memory_allocated() / (1024**2) # MB
        final_cached = torch.cuda.memory_reserved() / (1024**2) # MB

        print(f"[{func.__name__}] Final GPU memory allocated: {final_allocated:.2f} MB, cached: {final_cached:.2f} MB")
        print(f"[{func.__name__}] GPU memory change (allocated): {final_allocated - initial_allocated:.2f} MB")
        print(f"[{func.__name__}] GPU memory change (cached): {final_cached - initial_cached:.2f} MB")
        return result
    return wrapper
                 