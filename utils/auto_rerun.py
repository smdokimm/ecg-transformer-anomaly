import gc
import torch
import time
import os
import uuid

def auto_rerun(func, runs=50, delay=3):
    """
    Automatically reruns the given function up to `runs` times,
    clearing CUDA cache, collecting garbage, and resetting
    WANDB_RUN_ID between runs.
    """
    for run_idx in range(1, runs + 1):
        print(f"=== RUN {run_idx}/{runs} complete ===")
        func()
        if run_idx < runs:
            # Clear GPU cache and collect Python garbage
            torch.cuda.empty_cache()
            gc.collect()
            # Short pause to ensure resources are freed
            time.sleep(delay)
            # Force a new W&B run ID
            os.environ['WANDB_RUN_ID'] = uuid.uuid4().hex