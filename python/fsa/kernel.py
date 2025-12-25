from typing import Optional
from .instructions import *
from .tensor import MTile, STile, ATile
from .config import get_config, get_mem_manager


@dataclass
class Kernel:
    instructions: list[Instruction]  # 硬件指令序列
    input: list[MTile]  # 输入张量
    output: list[MTile] | MTile | None  # 输出张量


class KernelContext:
    def __init__(self):
        self.rows = get_config().sa_rows  # 脉动阵列行数(如16)
        self.cols = get_config().sa_cols  # 脉动阵列列数(如16)
        self.instructions: list[Instruction] = []  # 累积的指令

    def tile_row_addr(self, tile: ATile | STile) -> int:
        # on-chip SRAMs are not byte-addressed, they are row-addressed
        if isinstance(tile, STile):
            assert (
                tile.shape[-1] == self.rows  # 必须匹配脉动阵列行数
            ), f"Expected tile with {self.rows} rows, got {tile.shape[-1]} rows"
        else:
            assert (
                tile.shape[-1] == self.cols  # 必须匹配脉动阵列列数
            ), f"Expected tile with {self.cols} cols, got {tile.shape[-1]} cols"
        cols, itemsize = tile.shape[-1], tile.dtype.itemsize
        # 计算行号
        return tile.data_ptr // (cols * itemsize)

    def tile_stride(self, tile: ATile | STile) -> int:
        if isinstance(tile, STile):
            assert tile.shape[-1] == self.rows
        else:
            assert tile.shape[-1] == self.cols
        # stride[-2]是行间步长(元素数)
        # 除以每行元素数得到行间步长(行数)
        return tile.stride[-2] // tile.shape[-1]

    def push(self, inst: Instruction) -> None:
        self.instructions.append(inst)


__g_kernel_ctx: Optional[KernelContext] = None


def kernel(func):
    def wrapper(*args, **kwargs):
        global __g_kernel_ctx
        # 1. 创建新的编译上下文
        assert __g_kernel_ctx is None, "Nested kernels are not supported yet!"
        __g_kernel_ctx = KernelContext()
        # 2. 执行用户函数(收集指令)
        ret = func(*args, **kwargs)
        # 3. 验证返回值类型
        assert (
            (ret is None) or (isinstance(ret, MTile)) or (isinstance(ret, list))
        ), "the return type of FSA kernel function can only be one of MTile, list[MTile] or None"
        # 4. 创建Kernel对象
        kernel = Kernel(
            __g_kernel_ctx.instructions,  # 生成的指令
            get_mem_manager().mem_tensor_list,  # 所有输入
            ret,  # 输出
        )
        __g_kernel_ctx = None
        return kernel

    return wrapper


def check_kernel_ctx(func):
    def wrapper(*args, **kwargs):
        # 确保在@kernel装饰的函数内调用
        assert (
            __g_kernel_ctx is not None
        ), f"{func.__name__} can only be called within a FSA kernel!"
        func(*args, **kwargs)

    return wrapper


@check_kernel_ctx
def fence(mx: bool, dma: bool, stop: bool) -> None:
    # 添加屏障指令
    __g_kernel_ctx.push(FenceInstruction(mx, dma, stop))


@check_kernel_ctx
def dma(
    func: int,  # DMA操作类型
    mem: MTile,  # 主存tile
    tile: ATile | STile,  # 片上tile
    sem: Optional[Semaphore],  # 信号量
    aq: bool = True,  # 是否需要acquire
    rl: bool = True,  # 是否需要release
) -> None:
    assert mem.shape == tile.shape
    assert len(mem.shape) == 2
    assert mem.dtype == tile.dtype
    rows, cols = mem.shape
    # 主存步长可能超过指令字段能表示的范围
    mem_full_stride = mem.stride[-2] * mem.dtype.itemsize
    # check width
    inst_full_stride = InstructionField(mem_full_stride, 6 + 15 - 1, 0, signed=True)
    # 将32位步长拆分为两部分,放入2个指令字段
    # 高6位 → stride_1 (mem_stride_1)
    # 低15位 → stride_2 (stride_2)
    # split full stride into two parts
    # high 6 bits of full stride
    stride_1 = (inst_full_stride.value >> 15) & ((1 << 6) - 1)
    # low 15 bits of full stride
    stride_2 = inst_full_stride.value & ((1 << 15) - 1)

    mem = DMAInstrucionMem(mem.data_ptr, stride_2, cols * mem.dtype.itemsize)
    sram = DMAInstrucionSRAM(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        isAccum=isinstance(tile, ATile),
        mem_stride_1=stride_1,
    )
    if sem is None:
        aq, rl = False, False
    header = DMAInstructionHeader(
        sem.id if sem else 0,
        acquireValid=aq,
        acquireSemValue=sem.value if sem and aq else 0,
        releaseValid=rl,
        releaseSemValue=sem.inc().value if rl else 0,
        func=func,
        repeat=rows,
    )
    __g_kernel_ctx.push(DMAInstruction(header, sram, mem))


@check_kernel_ctx
def load_tile(
    mem: MTile, tile: STile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True
) -> None:
    # 主存 → 暂存器
    dma(DMAFunc.LD_SRAM.value, mem, tile, sem, aq, rl)


@check_kernel_ctx
def store_tile(
    tile: ATile, mem: MTile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True
) -> None:
    # 累加器 → 主存(经过归一化)
    dma(DMAFunc.ST_SRAM.value, mem, tile, sem, aq, rl)


def build_matrix_instruction_header(
    func: int, waitPrevAcc: bool, sem: Optional[Semaphore], aq: bool, rl: bool
) -> MatrixInstructionHeader:
    if sem is None:
        # 无信号量时关闭同步
        aq, rl = False, False

    return MatrixInstructionHeader(
        sem.id if sem else 0,
        acquireValid=aq,
        acquireSemValue=sem.value if sem and aq else 0,
        releaseValid=rl,
        releaseSemValue=sem.inc().value if rl else 0,  # 递增信号量
        func=func,
        waitPrevAcc=waitPrevAcc,  # 等待前一个累加完成
    )


@check_kernel_ctx
def mx_load_stationary(
    tile: STile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True
) -> None:
    # 将数据从暂存器加载到脉动阵列的固定位置
    # 用于加载Q矩阵(stationary模式)
    assert len(tile.shape) == 2 and tile.shape[-1] == __g_kernel_ctx.rows
    header = build_matrix_instruction_header(
        MxFunc.LOAD_STATIONARY.value, False, sem, aq, rl
    )
    spad = MatrixInstructionSpad(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        False,
        False,
        False,
    )
    acc = MatrixInstrucionAcc(0, 0, False)
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))


@check_kernel_ctx
def mx_attn_score(
    k: STile,  # K矩阵块(已加载到暂存器)
    l: ATile,  # L矩阵(行最大值/LogSumExp)
    accumulate: bool,  # 是否累加(在线softmax)
    sem: Optional[Semaphore],
    aq: bool = True,
    rl: bool = True,
) -> None:
    # 计算:S_ij = Q_i × K_jᵀ(Q已在脉动阵列中)
    assert len(k.shape) == 2 and l.shape == (1, __g_kernel_ctx.cols)
    header = build_matrix_instruction_header(
        MxFunc.ATTN_SCORE.value, False, sem, aq, rl
    )
    spad = MatrixInstructionSpad(
        __g_kernel_ctx.tile_row_addr(k), __g_kernel_ctx.tile_stride(k), True, True, True
    )
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(l), __g_kernel_ctx.tile_stride(l), not accumulate
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))


@check_kernel_ctx
def mx_attn_value(
    v_t: STile,
    o_t: ATile,
    accumulate: bool,
    sem: Optional[Semaphore],
    aq: bool = True,
    rl: bool = True,
) -> None:
    # 计算:O_i += softmax(S_ij) × V_j
    assert len(v_t.shape) == 2 and len(o_t.shape) == 2
    header = build_matrix_instruction_header(
        MxFunc.ATTN_VALUE.value, False, sem, aq, rl
    )
    spad = MatrixInstructionSpad(
        __g_kernel_ctx.tile_row_addr(v_t),
        __g_kernel_ctx.tile_stride(v_t),
        True,
        False,
        True,
    )
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(o_t),
        __g_kernel_ctx.tile_stride(o_t),
        not accumulate,
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))


@check_kernel_ctx
def mx_reciprocal(
    tile: ATile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True
) -> None:
    """计算倒数"""
    assert tile.shape == (1, __g_kernel_ctx.cols)
    header = build_matrix_instruction_header(
        MxFunc.ACC_RECIPROCOL.value, True, sem, aq, rl
    )
    spad = MatrixInstructionSpad(0, 0, False, False, False)
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(tile), __g_kernel_ctx.tile_stride(tile), False
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))


@check_kernel_ctx
def mx_attn_lse_norm(
    tile: ATile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True
) -> None:
    """LSE归一化"""
    assert len(tile.shape) == 2 and tile.shape[-1] == __g_kernel_ctx.cols
    header = build_matrix_instruction_header(
        MxFunc.ATTN_LSE_NORM.value, True, sem, aq, rl
    )
    spad = MatrixInstructionSpad(0, 0, False, False, False)
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(tile), __g_kernel_ctx.tile_stride(tile), False
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))
