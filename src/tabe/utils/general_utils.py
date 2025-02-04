from contextlib import contextmanager
import os
import sys

import numpy as np
import random
import torch


@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr  # Store original stderr
        try:
            sys.stdout = devnull  # Redirect stdout to devnull
            sys.stderr = devnull  # Redirect stderr to devnull (for tqdm)
            yield
        finally:
            sys.stdout = old_stdout  # Restore original stdout
            sys.stderr = old_stderr  # Restore original stderr


def set_all_seeds(seed=42):
    """
    Sets the seed for reproducibility for numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables benchmark mode for reproducibility
