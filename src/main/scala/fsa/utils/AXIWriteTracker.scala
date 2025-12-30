package fsa.utils

import chisel3._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.amba.axi4.AXI4BundleParameters
import chisel3.util.HasBlackBoxResource

// AXI4写事务跟踪器,用于监控和记录AXI4总线上的写操作, 可用于调试和验证.
class AXI4WriteTracker(params: AXI4BundleParameters)
    extends BlackBox(
      Map(
        "ADDR_BITS" -> params.addrBits,
        "SIZE_BITS" -> params.sizeBits,
        "LEN_BITS" -> params.lenBits,
        "DATA_BITS" -> params.dataBits
      )
    )
    with HasBlackBoxResource {
    val io = IO(new Bundle {
        // 时钟
        val clock = Input(Clock())

        // AW通道(写地址)
        val aw_fire = Input(Bool()) // aw.valid && aw.ready
        val aw_addr = Input(UInt(params.addrBits.W)) // 起始地址
        val aw_size = Input(UInt(params.sizeBits.W)) // 传输大小(字节)
        val aw_len = Input(UInt(params.lenBits.W)) // 突发长度

        // W通道(写数据)
        val w_fire = Input(Bool()) // w.valid && w.ready
        val w_data = Input(UInt(params.dataBits.W)) // 写数据
        val w_last = Input(Bool()) // 突发传输的最后一拍
    })

    addResource("/fsa/vsrc/AXI4WriteTracker.v")
    addResource("/fsa/csrc/AXI4WriteTracker.cc")
}
