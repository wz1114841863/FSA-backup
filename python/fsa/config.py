from .dtype import *
from .mem import CompoundMemoryManger
import json

"""
# FSA内存结构描述(示意图, 可能存在理解错误)
┌─────────────────────────────────┐
│         主内存 (256MB)           │
│  Q, K, V, 中间结果等             │
└──────────────┬──────────────────┘
               │ DMA传输
┌──────────────▼──────────────────┐
│    暂存器 (4KB)                  │
│    ┌──────────────┐             │
│    │ 当前Tile数据  │◄─┐          │
│    └──────────────┘  │          │
└──────────────┬───────┘          │
               │                  │
         ┌─────▼─────┐            │
         │ 脉动阵列   │            │
         │  16x16 PE │            │
         └─────┬─────┘            │
               │                  │
         ┌─────▼─────┐            │
         │ 累加器    │◄──────────┘
         │  (4KB)    │
         │ fp32精度  │
         └───────────┘

"""


@dataclass(frozen=True)
class FSAConfig:
    sa_rows: int = 16  # number of rows in systolic array
    sa_cols: int = 16  # number of columns in systolic array
    inst_queue_size: int = 256  # size of instruction queue
    # 数据类型配置
    e_type: dtype = fp16  # element type
    a_type: dtype = fp32  # accumulator type
    # 主内存(DDR/外部RAM)相关配置
    mem_base: int = 0x80000000  # 内存基地址
    mem_size: int = 0x10000000  # 内存大小:256MB (0x10000000 = 256×1024×1024)
    mem_align: int = 32  # 内存对齐要求(字节)
    # 暂存存储器(Scratchpad)相关配置
    spad_base: int = 0  # 暂存器基地址(片上SRAM)
    spad_size: int = 0x1000  # 大小:4KB (0x1000 = 4096 bytes)
    # 累加器存储器(Accumulator)相关配置
    acc_base: int = 0  # 累加器基地址(片上SRAM)
    acc_size: int = 0x1000  # 大小:4KB (0x1000 = 4096 bytes)


@dataclass(frozen=True)
class FSAGlobalVariables:
    config: FSAConfig
    mem_manager: CompoundMemoryManger


# 私有全局变量(前导双下划线表示模块私有), 单例模式
__global_vars: FSAGlobalVariables = None


def init(config_file: str):
    global __global_vars
    assert __global_vars is None, "FSA is already initialized."

    with open(config_file, "r") as f:
        cfg = json.load(f)
    cfg["e_type"] = eval(cfg["e_type"])  # fp16被定义
    cfg["a_type"] = eval(cfg["a_type"])  # fp32被定义
    config = FSAConfig(**cfg)
    mem_manger = CompoundMemoryManger(
        mem_base=config.mem_base,
        mem_size=config.mem_size,
        mem_align=config.mem_align,
        spad_base=config.spad_base,
        spad_size=config.spad_size,
        spad_align=config.sa_cols * config.e_type.itemsize,
        spad_dtype=config.e_type,
        acc_base=config.acc_base,
        acc_size=config.acc_size,
        acc_align=config.sa_cols * config.a_type.itemsize,
        acc_dtype=config.a_type,
    )
    __global_vars = FSAGlobalVariables(config, mem_manger)


def require_initialized():
    global __global_vars
    if __global_vars is None:
        raise RuntimeError(
            "FSA is not initialized. Call init() with a config file before using FSA."
        )
    return __global_vars


def get_config() -> FSAConfig:
    """Get the global FSA configuration."""
    require_initialized()
    return __global_vars.config


def get_mem_manager() -> CompoundMemoryManger:
    """Get the global FSA memory manager."""
    require_initialized()
    return __global_vars.mem_manager
