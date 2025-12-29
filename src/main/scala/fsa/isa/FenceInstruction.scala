package fsa.isa

import chisel3._
import ISA.Constants._

/*
    栅栏指令定义: 用于同步和控制执行流
    指令格式(32位):
位31 ┌─────────────────────────────────┐
     │              _pad               │ ← 最后定义,在最高位
位6  ├─────────────────────────────────┤
     │ stop                            │
位5  ├─────────────────────────────────┤
     │ dma                             │
位4  ├─────────────────────────────────┤
     │ matrix                          │
位3  ├─────────────────────────────────┤
     │          instType[2:0]          │ ← 最先定义,在最低位
位0  └─────────────────────────────────┘

 */

class FenceInstruction extends NBytesBundle(4) with HasInstructionType {
    val matrix = Bool() // 针对矩阵计算单元的栅栏
    val dma = Bool() // 针对DMA单元的栅栏
    val stop = Bool() // 是否停止执行
    val _pad = padOpt(I_TYPE_BITS + 3)
    checkWidth()
}
