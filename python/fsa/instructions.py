from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence
from enum import Enum

"""
┌─────────────────┐
│   主存 (DDR)     │
│                 │
└──────┬──────────┘
       │ DMA指令控制
┌──────▼──────────┐
│ 片上内存         │
│ ├─暂存器 (Spad)  │←─矩阵指令读取
│ └─累加器 (Acc)   │←─矩阵指令写入
│                 │
└──────┬──────────┘
       │ 脉动阵列访问
┌──────▼──────────┐
│  脉动阵列        │
│  16x16 PE       │
└─────────────────┘
"""


class InstructionType(Enum):
    # 指令类型
    FENCE = 0
    MX = 1
    DMA = 2


class MxFunc(Enum):
    # Matrix操作功能
    LOAD_STATIONARY = 0  # 加载数据到脉动阵列
    ATTN_SCORE = 1  # 计算Q×Kᵀ(注意力分数)
    ATTN_VALUE = 2  # 计算softmax(QKᵀ)×V
    ACC_RECIPROCOL = 3  # 计算倒数(用于softmax归一化)
    ATTN_LSE_NORM = 4  # LogSumExp归一化


class DMAFunc(Enum):
    # DMA操作功能
    LD_SRAM = 0  # 主存 → SRAM
    ST_SRAM = 1  # SRAM → 主存


@dataclass
class InstructionField:
    """位字段封装"""

    value: int | bool  # 字段值
    msb: int  # 最高位位置
    lsb: int  # 最低位位置
    signed: bool = False  # 是否有符号

    @property
    def width(self) -> int:
        return self.msb - self.lsb + 1

    def shifted_value(self) -> int:
        # 将值移位到正确位置, 并掩码确保位宽正确
        return (self.value & ((1 << self.width) - 1)) << self.lsb

    def __post_init__(self):
        if isinstance(self.value, bool):
            assert not self.signed, "Boolean fields cannot be signed"
            self.value = int(self.value)
        if self.signed:
            assert (
                -(1 << (self.width - 1)) <= self.value < (1 << (self.width - 1))
            ), f"Value {self.value} cannot be represented in {self.width} bits as signed"
        else:
            assert (
                0 <= self.value < (1 << self.width)
            ), f"Value {self.value} cannot be represented in {self.width} bits as unsigned"


class InstructionLike(ABC):
    """指令片段抽象"""

    def combine_fields(fs: Sequence[InstructionField]) -> int:
        bits = 0
        for f in fs:
            # OR操作合并所有字段
            bits |= f.shifted_value()
        return bits

    @property
    @abstractmethod
    def bits(self) -> int:
        pass


class Instruction(InstructionLike):

    @property
    @abstractmethod
    def i_type(self) -> InstructionType:
        # 返回指令类型
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        # 返回指令宽度(以位为单位)
        pass

    def to_ui32_list(self) -> list[int]:
        # 将长指令拆分为32位字(用于存储/传输)
        n_pieces = self.width // 32
        res = []
        bits = self.bits
        for _ in range(n_pieces):
            ui32 = bits & 0xFFFFFFFF  # 取低32位
            res.append(ui32)
            bits >>= 32  # 右移32位以处理下一块
        return res

    @property
    def n_bytes(self) -> int:
        # 返回指令宽度(以字节为单位)
        return self.width // 8


@dataclass
class FenceInstruction(Instruction):
    """屏障指令

    31-29: 指令类型 (FENCE=0)
    28:    mx屏障
    27:    dma屏障
    26:    stop标志
    25-0:  保留
    """

    mx: bool  # 等待矩阵操作完成
    dma: bool  # 等待DMA操作完成
    stop: bool  # 停止处理器

    @property
    def i_type(self) -> InstructionType:
        return InstructionType.FENCE

    @property
    def width(self) -> int:
        # 屏障指令宽度为32位
        return 32

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.i_type.value, 31, 29),
                InstructionField(self.mx, 28, 28),
                InstructionField(self.dma, 27, 27),
                InstructionField(self.stop, 26, 26),
            )
        )


@dataclass
class MatrixInstructionHeader(InstructionLike):
    """矩阵指令头部"""

    semId: int  # 信号量ID, 0-31
    acquireValid: bool  # 是否获取信号量
    acquireSemValue: int  # 获取的信号量值, 0-7
    releaseValid: bool  # 是否释放信号量
    releaseSemValue: int  # 释放的信号量值, 0-7
    func: int  # 操作类型, mxFunc枚举
    waitPrevAcc: bool  # 等待前一个累加器操作完成

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.semId, 28, 24),
                InstructionField(self.acquireValid, 23, 23),
                InstructionField(self.acquireSemValue, 22, 20),
                InstructionField(self.releaseValid, 19, 19),
                InstructionField(self.releaseSemValue, 18, 16),
                InstructionField(self.func, 15, 11),
                InstructionField(self.waitPrevAcc, 10, 10),
            )
        )


@dataclass
class MatrixInstructionSpad(InstructionLike):
    """暂存器访问"""

    addr: int  # 暂存器地址
    stride: int  # 数据步长(有符号)
    revInput: bool  # 反转输入顺序
    revOutput: bool  # 反转输出顺序
    delayOutput: bool  # 延迟输出

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.addr, 31, 12),
                InstructionField(self.stride, 11, 7, signed=True),
                InstructionField(self.revInput, 6, 6),
                InstructionField(self.revOutput, 5, 5),
                InstructionField(self.delayOutput, 4, 4),
            )
        )


@dataclass
class MatrixInstrucionAcc(InstructionLike):
    """累加器访问"""

    addr: int  # 累加器地址
    stride: int  # 数据步长(有符号)
    zero: bool  # 是否清零累加器

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.addr, 31, 12),
                InstructionField(self.stride, 11, 7, signed=True),
                InstructionField(self.zero, 6, 6),
            )
        )


@dataclass
class MatrixInstruction(Instruction):
    """完整的矩阵指令

    [95-64] Acc部分
    [63-32] Spad部分
    [31-0]  Header + 指令类型
    """

    header: MatrixInstructionHeader
    spad: MatrixInstructionSpad
    acc: MatrixInstrucionAcc

    @property
    def i_type(self) -> InstructionType:
        return InstructionType.MX

    @property
    def width(self) -> int:
        return 3 * 32

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.header.bits, 28, 0),
                InstructionField(self.i_type.value, 31, 29),
                InstructionField(self.spad.bits, 63, 32),
                InstructionField(self.acc.bits, 95, 64),
            )
        )


@dataclass
class DMAInstructionHeader(InstructionLike):
    """DMA指令头部"""

    semId: int  # 信号量ID
    acquireValid: bool  # 是否获取信号量
    acquireSemValue: int  # 获取的信号量值
    releaseValid: bool  # 是否释放信号量
    releaseSemValue: int  # 释放的信号量值
    func: int  # DMA操作类型, DMAFunc枚举
    repeat: int  # 重复次数

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.semId, 28, 24),
                InstructionField(self.acquireValid, 23, 23),
                InstructionField(self.acquireSemValue, 22, 20),
                InstructionField(self.releaseValid, 19, 19),
                InstructionField(self.releaseSemValue, 18, 16),
                InstructionField(self.func, 15, 12),
                InstructionField(self.repeat, 11, 3),
            )
        )


@dataclass
class DMAInstrucionSRAM(InstructionLike):
    """SRAM访问"""

    addr: int  # SRAM地址
    stride: int  # 步长
    isAccum: bool  # 是否为累加器(否则是暂存器)
    mem_stride_1: int  # 内存步长1

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.addr, 31, 12),
                InstructionField(self.stride, 11, 7, signed=True),
                InstructionField(self.isAccum, 6, 6),
                InstructionField(self.mem_stride_1, 5, 0),
            )
        )


@dataclass
class DMAInstrucionMem(InstructionLike):
    """内存访问"""

    addr: int  # 主存地址
    stride_2: int  # 内存步长2
    size: int  # 传输大小

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.addr, 63, 25),
                InstructionField(self.stride_2, 24, 10),
                InstructionField(self.size, 9, 0),
            )
        )


@dataclass
class DMAInstruction(Instruction):
    """完整的DMA指令

    [127-64] Mem部分
    [63-32] SRAM部分
    [31-0]  Header + 指令类型

    """

    header: DMAInstructionHeader
    sram: DMAInstrucionSRAM
    mem: DMAInstrucionMem

    @property
    def i_type(self) -> InstructionType:
        return InstructionType.DMA

    @property
    def width(self) -> int:
        return 4 * 32

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields(
            (
                InstructionField(self.header.bits, 28, 0),
                InstructionField(self.i_type.value, 31, 29),
                InstructionField(self.sram.bits, 63, 32),
                InstructionField(self.mem.bits, 127, 64),
            )
        )


class Semaphore:
    """信号量

    实现指令间的依赖和同步,避免数据竞争
    """

    def __init__(self, id: int, n: int):
        assert 0 <= id < 32 and 0 < n < 8
        self.id = id  # 信号量ID (0-31)
        self.n = n  # 信号量最大值 (1-7)
        self.value = 0  # 当前值

    def inc(self) -> "Semaphore":
        if self.value == self.n - 1:
            self.value = 0
        else:
            self.value += 1
        return self
