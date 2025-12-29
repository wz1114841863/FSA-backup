package fsa.isa

import chisel3._
import chisel3.util._
import freechips.rocketchip.util.UIntIsOneOf

import ISA.Constants._

/*
    精简的特定指令集架构定义, 指令集分为三类:
    1. FENCE指令: 用于同步和内存屏障
    2. MATRIX指令: 用于控制脉动阵列的矩阵
    3. DMA指令: 用于控制直接内存访问操作
 */

// 所有指令共有的类型字段
trait HasInstructionType { this: Bundle => // 自类型标注
    val instType = UInt(I_TYPE_BITS.W)
}

// 实现硬件级别的同步原语
trait HasSemaphore { this: Bundle =>
    val semId = UInt(SEM_ID_BITS.W) // 信号量ID(5bit, 支持32个信号量)
    val acquireValid = Bool() // 是否获取信号量
    val acquireSemValue = UInt(SEM_VALUE_BITS.W) // 获取信号量的值
    val releaseValid = Bool() // 是否释放信号量
    val releaseSemValue = UInt(SEM_VALUE_BITS.W) // 释放信号量的值
    // 信号量相关字段的总位宽
    def semBits: Int = SEM_ID_BITS + 2 * (1 + SEM_VALUE_BITS)
}

// 地址空间支持
trait HasAddr { this: Bundle =>
    def addrWidth: Int // 实际地址宽度
    def maxAddrWidth: Int // 最大地址宽度
    val _pad_addr_msb = // 地址对齐填充
        if (maxAddrWidth > addrWidth) Some(UInt((maxAddrWidth - addrWidth).W))
        else None
    val addr = UInt(addrWidth.W) // 地址值
}

// 确保指令长度是字节对齐的, 便于指令存储和取值
abstract class NBytesBundle(n: Int) extends Bundle {
    def checkWidth(): Unit = {
        require(this.getWidth == n * 8, f"width: ${this.getWidth} n: $n")
    }
    def padOpt(existingBits: Int) = if (existingBits == n * 8) None
    else Some(UInt((n * 8 - existingBits).W))
}

object ISA {

    // 常量定义
    object Constants {
        // 指令类型:3位,8种类型
        val I_TYPE_BITS = 3
        // 信号量:32个,5位ID,3位值(0-7)
        val N_SEMAPHORES = 32
        val SEM_ID_BITS = log2Up(N_SEMAPHORES)
        val SEM_VALUE_BITS = 3

        // 矩阵指令功能:5位,32种功能
        val MX_FUNC_BITS = 5
        // ScratchPad(片上缓存)地址空间
        val SPAD_MAX_ADDR_BITS = 20 // 1MB地址空间, 2^20数据宽度
        val SPAD_STRIDE_BITS = 5
        // Accumulator地址空间
        val ACC_MAX_ADDR_BITS = 20 // 1MB地址空间, 2^20数据宽度
        val ACC_STRIDE_BITS = 5
        // 统一为最大宽度,简化设计
        val SRAM_MAX_ADDR_BITS = Seq(SPAD_MAX_ADDR_BITS, ACC_MAX_ADDR_BITS).max
        val SRAM_STRIDE_BITS = Seq(SPAD_STRIDE_BITS, ACC_STRIDE_BITS).max

        // DMA参数
        val DMA_FUNC_BITS = 4 // DMA功能: 4位,16种功能
        val DMA_SIZE_BITS = 10 // 传输大小: 10位, 最大1024个数据单元
        val DMA_REPEAT_BITS = 9 // 传输重复次数: 9位, 最大512次
        // 外部内存地址空间
        val MEM_MAX_ADDR_BITS = 39 // 512GB地址空间, 2^39数据宽度
        val MEM_STRIDE_1_BITS = 6
        val MEM_STRIDE_2_BITS = 15
        val MEM_STRIDE_BITS = MEM_STRIDE_1_BITS + MEM_STRIDE_2_BITS
    }

    // 指令类型枚举
    object InstTypes {
        val FENCE = 0
        val MATRIX = 1
        val DMA = 2
    }

    // 矩阵指令功能码
    object MxFunc {
        def LOAD_STATIONARY = 0.U // 加载静止矩阵(如Q)
        def ATTENTION_SCORE_COMPUTE = 1.U // 计算S = Q @ K^T + Softmax
        def ATTENTION_VALUE_COMPUTE = 2.U // 计算O = P @ V
        def ATTENTION_LSE_NORM_SCALE = 3.U // LSE归一化(缩放因子)
        def ATTENTION_LSE_NORM = 4.U // LSE归一化
    }

    // DMA指令功能码
    object DMAFunc {
        def LD_SRAM = 0.U // 从主存加载到SRAM
        def ST_SRAM = 1.U // 从SRAM存储到主存
    }

}
