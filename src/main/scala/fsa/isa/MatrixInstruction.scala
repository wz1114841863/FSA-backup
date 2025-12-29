package fsa.isa

import chisel3._
import ISA.Constants._

class MatrixInstructionHeader
    extends NBytesBundle(4)
    with HasInstructionType // 包含instType字段
    with HasSemaphore { // 包含信号量相关字段
    val func = UInt(MX_FUNC_BITS.W) // 5位功能码
    val waitPrevAcc = Bool() // 1位, 等待前一个累加器操作完成
    val _pad = padOpt(I_TYPE_BITS + semBits + MX_FUNC_BITS + 1)
    checkWidth()
}

class MatrixInstructionSpad(val addrWidth: Int)
    extends NBytesBundle(4)
    with HasAddr { // 包含地址字段
    val stride = SInt(SPAD_STRIDE_BITS.W) // 5位有符号步长
    val revInput = Bool() // 是否反转输入
    val revOutput = Bool() // 是否反转输出
    val delayOutput = Bool() // 是否延迟输出
    val _pad = padOpt(SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 3)
    override def maxAddrWidth: Int = SPAD_MAX_ADDR_BITS // 20位
    checkWidth()
}

class MatrixInstructionAcc(val addrWidth: Int)
    extends NBytesBundle(4)
    with HasAddr { // 包含地址字段
    val stride = SInt(ACC_STRIDE_BITS.W) // 5位有符号步长
    val zero = Bool()
    val _pad = padOpt(ACC_MAX_ADDR_BITS + ACC_STRIDE_BITS + 1)
    override def maxAddrWidth: Int = ACC_MAX_ADDR_BITS
    checkWidth()
}

/*
位95 ┌────────────────────────────────┐
     │    header部分 (32位)           │
位64 ├────────────────────────────────┤
     │    spad部分 (32位)             │
位32 ├────────────────────────────────┤
     │    acc部分 (32位)              │
位0  └────────────────────────────────┘
 */
class MatrixInstruction(spAddrWidth: Int, accAddrWidth: Int)
    extends NBytesBundle(12) {
    val acc = new MatrixInstructionAcc(accAddrWidth)
    val spad = new MatrixInstructionSpad(spAddrWidth)
    val header = new MatrixInstructionHeader
    checkWidth()
}
