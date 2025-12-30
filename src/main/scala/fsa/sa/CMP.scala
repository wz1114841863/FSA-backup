package fsa.sa

import chisel3._
import chisel3.util._
import fsa.arithmetic._
import fsa.arithmetic.ArithmeticSyntax._

object CmpControlCmd {
    def width = 3 // 3位, 8种命令
    def UPDATE = 0.U // 更新最大值
    def PROP_MAX = 1.U // 传播最大值
    def PROP_MAX_DIFF = 2.U // 传播最大值差值
    def PROP_ZERO = 3.U // 传播零值
    def RESET = 4.U // 重置
    def PROP_EXP2_INTERCEPTS = 5.U // 传播exp2分段截距
}

class CmpControl extends Bundle {
    val cmd = UInt(CmpControlCmd.width.W)
}

class CMP[E <: Data: Arithmetic, A <: Data: Arithmetic](
    ev: ArithmeticImpl[E, A] // 算术实现参数
) extends Module {
    val (accType, cmpUnitGen) = (ev.accType, ev.accCmp _)
    val io = IO(new Bundle {
        val d_input = Flipped(Valid(accType)) // 数据输入
        val d_output = Valid(accType) // 数据输出
        val in_ctrl = Flipped(Valid(new CmpControl)) // 控制输入
        val out_ctrl = Valid(new CmpControl) // 控制输出
    })

    val cmpUnit = Module(cmpUnitGen())

    val oldMax = RegInit(accType.minimum) // 旧的最大值(初始为-∞)
    val newMax = RegInit(accType.minimum) // 新的最大值(初始为-∞)

    val cmd = io.in_ctrl.bits.cmd
    val update_new_max = cmd === CmpControlCmd.UPDATE // 更新最大值
    val prop_new_max = cmd === CmpControlCmd.PROP_MAX // 传播最大值
    val prop_diff = cmd === CmpControlCmd.PROP_MAX_DIFF // 传播差值
    val prop_zero = cmd === CmpControlCmd.PROP_ZERO // 传播零值
    val do_reset = cmd === CmpControlCmd.RESET // 重置
    val prop_exp2_intercepts =
        cmd === CmpControlCmd.PROP_EXP2_INTERCEPTS // 传播exp2分段截距
    val zero = accType.zero

    // 比较单元输入选择
    cmpUnit.io.in_a := Mux(
      update_new_max,
      io.d_input.bits, // 情况1: 使用输入数据更新最大值
      Mux(prop_new_max, zero, oldMax) // 情况2: 传播最大值用零,否则用oldMax
    )
    cmpUnit.io.in_b := newMax // 总是与newMax比较

    val exp2_intercepts = VecInit(ev.exp2PwlIntercepts)
    val exp2_counter = Counter(exp2_intercepts.length)
    when(io.in_ctrl.fire && prop_exp2_intercepts) {
        exp2_counter.inc()
    }

    when(io.in_ctrl.fire) { // 当控制有效且准备好时
        when(do_reset) {
            newMax := accType.minimum // 重置为负无穷
            oldMax := accType.minimum
        }.elsewhen(prop_zero || prop_exp2_intercepts) {
            // do nothing, keep newMax and oldMax unchanged
        }.otherwise({
            newMax := cmpUnit.io.out_max // 更新newMax为比较结果的最大值
            when(prop_diff) {
                oldMax := cmpUnit.io.out_max // 如果是传播差值,更新oldMax
            }
        })
    }

    val downCastDIn = ev.viewEasA(ev.cvtAtoE(io.d_input.bits))
    io.d_output.bits := Mux(
      prop_zero,
      zero, // 如果是传播零值,输出零
      Mux(
        prop_exp2_intercepts,
        exp2_intercepts(exp2_counter.value), // 分发截距
        Mux(update_new_max, downCastDIn, cmpUnit.io.out_diff) // 更新时传输入,否则传差值
      )
    )
    io.d_output.valid := io.in_ctrl.valid && !do_reset // 重置时不输出有效数据
    io.out_ctrl := io.in_ctrl // 控制信号直接传递
}
