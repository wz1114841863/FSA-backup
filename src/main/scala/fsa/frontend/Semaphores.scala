package fsa.frontend

import chisel3._
import chisel3.util._
import fsa.isa.ISA.Constants._

/*
信号量控制流程:
    发射前:检查acquireValid,等待信号量就绪
    发射时:标记信号量为忙(防止其他指令获取)
    执行中:保持忙状态
    完成后:根据releaseValid更新信号量值并清除忙状态
 */
class Semaphore extends Bundle {
    val id = UInt(SEM_ID_BITS.W) // 5位, 支持32个信号量
    val value = UInt(SEM_VALUE_BITS.W) // 3位, 值范围0-7
}

class Semaphores(nRead: Int, nWrite: Int) extends Module {
    val io = IO(new Bundle {
        val acquire =
            Vec(nRead, Flipped(Decoupled(new Semaphore))) // 获取请求(握手接口)
        val release = Vec(nWrite, Flipped(Valid(new Semaphore))) // 释放请求(有效接口)
    })

    val semaphores = RegInit(VecInit(Seq.fill(N_SEMAPHORES) {
        0.U(SEM_VALUE_BITS.W) // 32个信号量,每个3位,初始值为0
    }))
    // 每个信号量一个忙标志,防止重复获取
    val busy = RegInit(VecInit(Seq.fill(N_SEMAPHORES) { false.B }))

    io.release.foreach { release =>
        when(release.fire) { // release.fire = release.valid(Valid接口的特性)
            busy(release.bits.id) := false.B // 清除忙状态
            semaphores(release.bits.id) := release.bits.value // 更新信号量值
        }
    }

    // 两个条件必须同时满足:
    // 1. 信号量不忙(!busy(id))
    // 2. 请求值等于当前值(value === semaphores(id))
    io.acquire.foreach { acq =>
        acq.ready := !busy(acq.bits.id) && acq.bits.value === semaphores(
          acq.bits.id
        )
        when(acq.fire) {
            // 标记为忙,防止其他请求获取
            busy(acq.bits.id) := true.B
        }
    }

}
