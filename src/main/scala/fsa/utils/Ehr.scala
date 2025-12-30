package fsa.utils

import chisel3._
import chisel3.util._

// Bluespec-style Ephemeral History Registers
class Ehr[T <: Data](n: Int, gen: T, init: Option[T]) extends Module {

    val io = IO(new Bundle {
        val read = Output(Vec(n, gen)) // n个读端口
        val write = Flipped(Vec(n, Valid(gen))) // n个写端口
    })

    val reg = init.map(i => RegInit(i)).getOrElse(Reg(gen))

    io.read.head := reg // 第0个读端口看到的是当前寄存器值
    // 为 read(1)..read(n-1) 创建"时间旅行"视图
    io.read.tail.zip(io.write.init).foldLeft(reg) { case (r_last, (r, w)) =>
        // r 是当前读端口(如 read(1))
        // w 是对应的写端口(如 write(0))
        // r_last 是前一个读端口看到的值

        // 关键:如果写有效,读端口看到新值
        //      如果写无效,读端口看到前一个读端口的值
        r := Mux(w.valid, w.bits, r_last)
        r
    }
    // 高索引优先(reduceRight从右开始)
    val w = io.write.reduceRight((l, r) => Mux(r.valid, r, l))
    reg := Mux(w.valid, w.bits, reg)

    def read(idx: Int): T = io.read(idx)
    def write(idx: Int, v: T): Unit = {
        io.write(idx).valid := true.B
        io.write(idx).bits := v
    }

}

object Ehr {
    def apply[T <: Data](n: Int, gen: T, init: Option[T] = None) = {
        val ehr = Module(new Ehr(n, gen, init))
        ehr.io.write.foreach { w =>
            w.valid := false.B
            w.bits := DontCare
        }
        ehr
    }
}
