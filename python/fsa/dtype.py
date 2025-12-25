import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class dtype:
    # 唯一属性:字节大小
    itemsize: int


# 预定义数据类型
fp32 = dtype(4)  # 4字节 = 32位
fp16 = dtype(2)  # 2字节 = 16位
bf16 = dtype(2)  # 2字节 = 16位(与fp16大小相同但格式不同)
fp8 = dtype(1)  # 1字节 = 8位


def get_dtype(ew: int, mw: int) -> dtype:
    match (ew, mw):
        case (8, 23):  # IEEE 754 float32: 8位指数,23位尾数
            return fp32
        case (8, 7):  # bfloat16: 8位指数,7位尾数
            return bf16
        case (5, 10):  # IEEE 754 float16: 5位指数,10位尾数
            return fp16
        case (4, 3):  # 自定义fp8: 4位指数,3位尾数
            return fp8
        case _:
            raise ValueError(f"Unknown dtype: e{ew}m{mw}")


def from_numpy_dtype(n_type: np.dtype):
    """Convert a NumPy dtype to the corresponding FSA dtype."""
    info = np.finfo(n_type)  # 获取浮点数信息
    return get_dtype(info.nexp, info.nmant)  # 转换指数/尾数位


def to_numpy_dtype(t: dtype):
    type_dict = {
        fp32: np.float32,
        fp16: np.float16,
    }
    return type_dict[t]
