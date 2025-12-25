from .tensor import STile, ATile, MTile, T
from .dtype import *
from typing import Type

"""
┌─────────────────┐
│   主内存 (DDR)   │ ← MemoryAllocator(mem_base, mem_size)
│   大容量/慢速    │   存放输入/输出数据/大矩阵
├─────────────────┤
│  暂存器 (Scratchpad) │ ← MemoryAllocator(spad_base, spad_size)
│  小容量/快速     │   存放当前计算的tile数据
├─────────────────┤
│  累加器 (Accumulator)│ ← MemoryAllocator(acc_base, acc_size)
│  中等容量/高精度  │   存放中间累加结果
└─────────────────┘

"""


class MemoryAllocator:
    def __init__(self, addr_base: int, size: int, alignment: int):
        self.addr_base = addr_base  # 内存区域起始地址
        self.size = size  # 内存区域总大小
        self.alignment = alignment  # 对齐要求
        # 空闲块列表: (起始地址, 大小)
        self.free_blocks = [(addr_base, size)]  # List of tuples (address, size)
        # 已分配块字典: {起始地址: 大小}
        self.allocated_blocks = {}

    def _align_up(self, addr: int) -> int:
        """Aligns the address to the nearest multiple of alignment."""
        if addr % self.alignment == 0:
            return addr
        return ((addr // self.alignment) + 1) * self.alignment

    def allocate(self, size: int) -> int:
        """Allocates a block of memory with the specified size."""
        # 遍历所有空闲块
        for index, (start, block_size) in enumerate(self.free_blocks):
            # 1. 对齐起始地址
            aligned_start = self._align_up(start)
            padding = aligned_start - start  # 为了满足对齐 需要的填充
            total_size = size + padding
            # 2. 检查是否有足够空间
            if total_size <= block_size:
                # Allocate the block
                self.free_blocks.pop(index)  # 移除空闲块
                allocated_addr = aligned_start
                self.allocated_blocks[allocated_addr] = size  # 记录已分配块

                # Split remaining free memory
                remaining_size = block_size - total_size
                if remaining_size > 0:
                    # 插入剩余的空闲块
                    self.free_blocks.insert(
                        index, (aligned_start + size, remaining_size)
                    )

                # print(f"Allocated {size} bytes at address {hex(allocated_addr)}")
                # 返回分配的地址
                return allocated_addr

        raise RuntimeError(
            f"Allocation failed: not enough memory. Requested {size} bytes, but only {self.size - sum(self.allocated_blocks.values())} bytes available."
        )

    def deallocate(self, addr: int):
        """Deallocates a previously allocated block of memory."""
        if addr in self.allocated_blocks:
            size = self.allocated_blocks.pop(addr)
            self.free_blocks.append((addr, size))
            self.free_blocks.sort()
            self._merge_free_blocks()
            # print(f"Deallocated {size} bytes from address {hex(addr)}")
        else:
            raise RuntimeError(f"Invalid deallocation attempt at address {hex(addr)}")

    def _merge_free_blocks(self):
        """Merges contiguous free blocks to prevent fragmentation."""
        merged_blocks = []
        for block in sorted(self.free_blocks):
            if (
                merged_blocks
                and merged_blocks[-1][0] + merged_blocks[-1][1] == block[0]
            ):
                # 与前一个块连续,合并
                last_addr, last_size = merged_blocks.pop()
                merged_blocks.append((last_addr, last_size + block[1]))
            else:
                # 不连续,添加新块
                merged_blocks.append(block)
        self.free_blocks = merged_blocks

    def dump_memory(self):
        """Prints the current memory layout."""
        print("\nAllocated Blocks:")
        for addr, size in self.allocated_blocks.items():
            print(f" - Address: {hex(addr)}, Size: {size} bytes")

        print("\nFree Blocks:")
        for addr, size in self.free_blocks:
            print(f" - Address: {hex(addr)}, Size: {size} bytes")
        print("\n")


class CompoundMemoryManger:
    """三级内存的统一管理器,为不同类型的tile提供专用分配接口"""

    def __init__(
        self,
        mem_base: int,  # 主内存起始地址
        mem_size: int,  # 主内存大小
        mem_align: int,  # 主内存对齐要求
        spad_base: int,  # 暂存器起始地址
        spad_size: int,  # 暂存器大小
        spad_align: int,  # 暂存器对齐要求
        spad_dtype: dtype,  # 暂存器数据类型(通常fp16)
        acc_base: int,  # 累加器起始地址
        acc_size: int,  # 累加器大小
        acc_align: int,  # 累加器对齐要求
        acc_dtype: dtype,  # 累加器数据类型(通常fp32)
    ):
        self.mem = MemoryAllocator(mem_base, mem_size, mem_align)
        self.spad = MemoryAllocator(spad_base, spad_size, spad_align)
        self.acc = MemoryAllocator(acc_base, acc_size, acc_align)
        self.spad_dtype = spad_dtype
        self.acc_dtype = acc_dtype
        self.mem_tensor_list: list[MTile] = []

    def alloc_spad(self, shape: int | tuple[int, ...]) -> STile:
        """分配暂存器内存(快速/小容量)"""
        return self.__allocate(self.spad, shape, self.spad_dtype, STile)

    def alloc_accumulator(self, shape: int | tuple[int, ...]) -> ATile:
        """分配累加器内存(高精度/中等容量)"""
        return self.__allocate(self.acc, shape, self.acc_dtype, ATile)

    def alloc_mem(self, shape: int | tuple[int, ...], dtype: dtype) -> MTile:
        """分配主内存(大容量/可自定义类型)"""
        tile = self.__allocate(self.mem, shape, dtype, MTile)
        self.mem_tensor_list.append(tile)
        return tile

    @staticmethod
    def __allocate(
        allocator: MemoryAllocator,  # 哪个分配器(mem/spad/acc)
        shape: int | tuple[int, ...],  # 形状
        dtype: dtype,  # 数据类型
        ret: Type[T],  # 返回类型(MTile/STile/ATile)
    ) -> T:
        # 1. 计算需要的内存大小(字节),  分配内存地址
        data_ptr = allocator.allocate(
            CompoundMemoryManger.__shape_to_size(shape) * dtype.itemsize
        )
        # 2. 返回对应类型的张量对象
        if isinstance(shape, int):
            shape = tuple(shape)
        return ret(shape, dtype, data_ptr)

    @staticmethod
    def __shape_to_size(shape: int | tuple[int, ...]) -> int:
        """计算形状对应的元素数量"""
        if isinstance(shape, int):
            if shape <= 0:
                raise ValueError(f"Shape dimension must be positive, got {shape}")
            return shape
        elif isinstance(shape, tuple):
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise ValueError(
                    f"All shape dimensions must be positive integers, got {shape}"
                )
            size = 1
            for dim in shape:
                size *= dim
            return size
        else:
            raise TypeError(f"Shape must be an int or tuple of ints, got {type(shape)}")
