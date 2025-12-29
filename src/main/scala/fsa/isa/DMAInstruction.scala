package fsa.isa

import chisel3._
import ISA.Constants._

class DMAInstructionHeader
    extends NBytesBundle(4)
    with HasInstructionType // instType字段
    with HasSemaphore { // 信号量相关字段
    val func = UInt(DMA_FUNC_BITS.W) // 4位功能码
    val repeat = UInt(DMA_REPEAT_BITS.W) // 9位重复次数
    val _pad = padOpt(I_TYPE_BITS + semBits + DMA_FUNC_BITS + DMA_REPEAT_BITS)
    checkWidth()
}

class DMAInstructionSRAM(val addrWidth: Int)
    extends NBytesBundle(4)
    with HasAddr { // 地址字段
    val stride = SInt(SRAM_STRIDE_BITS.W) // 5位步长
    val isAccum = Bool() // 是否为累加器地址空间
    val mem_stride1 = UInt(MEM_STRIDE_1_BITS.W) // 6位内存步长低部分
    val _pad = padOpt(
      SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + 1 + MEM_STRIDE_1_BITS
    )
    override def maxAddrWidth: Int = SRAM_MAX_ADDR_BITS
    checkWidth()
}

class DMAInstructionMem(val addrWidth: Int)
    extends NBytesBundle(8)
    with HasAddr { // 地址字段
    val stride2 = UInt(MEM_STRIDE_2_BITS.W) // 15位内存步长高部分
    val size = UInt(DMA_SIZE_BITS.W) // 10位传输大小
    val _pad = padOpt(MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS + DMA_SIZE_BITS)
    override def maxAddrWidth: Int = MEM_MAX_ADDR_BITS
    checkWidth()
}

/*
位127┌─────────────────────────────┐
     │      header (32位)         │ ← 最高32位
位96 ├─────────────────────────────┤
     │       sram (32位)          │
位64 ├─────────────────────────────┤
     │       mem (64位)           │ ← 最低64位
位0  └─────────────────────────────┘
 */
class DMAInstruction(sramAddrWidth: Int, memAddrWidth: Int)
    extends NBytesBundle(16) {
    val mem = new DMAInstructionMem(memAddrWidth)
    val sram = new DMAInstructionSRAM(sramAddrWidth)
    val header = new DMAInstructionHeader
    checkWidth()

    def getStride: SInt = (sram.mem_stride1 ## mem.stride2).asSInt
}
