package fsa.utils

import chisel3._
import chisel3.experimental.SourceInfo

// 延迟断言, 用于在断言失败后继续运行一段时间以捕获更多波形信息.
object DelayedAssert {
    // delay the `cond` to get more waveforms after the error occurs
    def apply(cond: Bool, delay: Int = 2)(implicit
        sourceInfo: SourceInfo
    ): assert.Assert = {
        assert((0 until delay).foldLeft(cond) { (c, _) =>
            RegNext(c, init = true.B)
        })
    }
}
