import os
import numpy as np
def find_latest_checkpoint(save_dir, prefix, suffix):
    all_files = os.listdir(save_dir)
    checkpoint_files = [i for i in all_files if i.startswith(prefix) and i.endswith(suffix)]
    checkpoint_num = [int(i.replace(prefix, "").replace(suffix, "")) for i in checkpoint_files]
    return max(checkpoint_num)
