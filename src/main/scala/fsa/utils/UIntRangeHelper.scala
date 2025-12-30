package fsa.utils

import chisel3._

// 为Chisel的UInt类型添加了两个便捷方法, 用于范围检查和相等性检查.
object UIntRangeHelper {
    implicit class UIntRangeHelper(x: UInt) {
        def between(start_inclusive: Int, end_exclusive: Int): Bool =
            start_inclusive.U <= x && x < end_exclusive.U
        def at(y: Int): Bool = x === y.U
    }
}
