from .dtype import dtype
from typing import Generic, TypeVar, Optional

T = TypeVar("T", bound="BaseTensor")

"""
硬件内存映射

┌─────────────────┐
│  主内存 (DDR)    │ ← MTile(大数据存储)
│  256MB-几GB      │
└────────┬────────┘
         │ DMA传输
┌────────▼────────┐
│ 片上暂存器      │ ← STile(计算tile)
│  4-64KB         │   低延迟访问
└────────┬────────┘
         │ 脉动阵列访问
┌────────▼────────┐
│  累加器         │ ← ATile(累加结果)
│  4-16KB         │   高精度存储
└─────────────────┘
"""


class BaseTensor(Generic[T]):

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: dtype,
        data_ptr: int = 0,
        stride: list[int] = None,
    ):
        self._shape: tuple[int, ...] = shape  # 张量形状
        self.dtype = dtype  # 数据类型
        self.data_ptr: int = data_ptr  # 内存起始地址
        self.stride: list[int] = (
            stride if stride is not None else self._calculate_stride(shape)
        )  # 步长数组,控制内存访问

    @staticmethod
    def _calculate_stride(shape: tuple[int, ...]) -> list[int]:
        """Calculate the stride for a contiguous memory layout."""
        stride = [1]  # 步长数组初始化,最后一个维度的步长为1
        for size in reversed(shape[1:]):
            stride.insert(0, stride[0] * size)
        return stride

    def is_contiguous(self) -> bool:
        """Check if the tensor's memory layout is contiguous."""
        expected_stride = self._calculate_stride(self._shape)
        return self.stride == expected_stride

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        """data size in bytes"""
        s = 1
        for dim in self.shape:
            s *= dim
        s *= self.dtype.itemsize
        return s

    def split(self, split_size: int, dim: int) -> tuple[T, ...]:
        """Split the tensor into chunks of split_size along the specified dimension."""
        assert (
            self.is_contiguous()
        ), "Currently only support splitting contiguous memory"

        dim_size = self._shape[dim]
        num_splits = (dim_size + split_size - 1) // split_size

        sub_tensors: list[T] = []

        for i in range(num_splits):
            # Calculate the starting data pointer offset
            offset = i * split_size * self.stride[dim] * self.dtype.itemsize

            # New shape for the sub-tensor
            new_shape = list(self._shape)
            new_shape[dim] = min(split_size, dim_size - i * split_size)

            sub_tensors.append(
                # 分割时不复制数据, 只创建新的张量对象指向不同的内存区域.
                type(self)(
                    shape=tuple(new_shape),
                    dtype=self.dtype,
                    data_ptr=self.data_ptr + offset,
                    stride=self.stride,
                )
            )

        return tuple(sub_tensors)

    def reverse(self, dim: int) -> T:
        """
        Reverse the tensor along the specified dimension.

        Args:
            dim (int): The dimension to reverse.

        Returns:
            T: A new instance of the tensor with the specified dimension reversed.
        """
        # Ensure the dimension is valid
        assert (
            0 <= dim < len(self._shape)
        ), f"Dimension {dim} out of range for tensor of shape {self._shape}"

        # Calculate the new stride
        new_stride = list(self.stride)
        new_stride[dim] *= -1

        # Calculate the new data pointer start position
        offset = (self._shape[dim] - 1) * abs(self.stride[dim]) * self.dtype.itemsize

        # Return a new tensor instance
        return type(self)(
            shape=self._shape,
            dtype=self.dtype,
            data_ptr=self.data_ptr + offset,
            stride=new_stride,
        )


class MTile(BaseTensor["MTile"]):
    # 可以存储实际数据
    data: Optional[bytes]

    def __init__(self, shape, dtype, data_ptr=0, stride=None):
        super().__init__(shape, dtype, data_ptr, stride)
        self.data = None


class STile(BaseTensor["STile"]):
    def __init__(self, shape, dtype, data_ptr=0, stride=None):
        super().__init__(shape, dtype, data_ptr, stride)


class ATile(BaseTensor["ATile"]):
    def __init__(self, shape, dtype, data_ptr=0, stride=None):
        super().__init__(shape, dtype, data_ptr, stride)
