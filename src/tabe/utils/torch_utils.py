import torch

torch_dtype_map = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def str_to_torch_dtype(dtype_str: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype_str, torch.dtype):
        return dtype_str  # Return as-is if it's already a torch dtype
    try:
        return torch_dtype_map[dtype_str]
    except KeyError:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
