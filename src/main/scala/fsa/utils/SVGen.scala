package fsa.utils

import circt.stage._
import fsa.Configs

// 将Chisel设计的SystolicArray模块转换为SystemVerilog代码
object SVGen {
    def main(args: Array[String]): Unit = {
        val rows = 128
        val cols = 128
        implicit val ev = Configs.fp16MulFp32AddArithmeticImpl
        ChiselStage.emitSystemVerilogFile(
          new fsa.sa.SystolicArray(rows, cols),
          args
        )
    }
}
