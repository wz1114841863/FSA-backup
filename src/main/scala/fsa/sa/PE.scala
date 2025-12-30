package fsa.sa

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy._
import fsa.arithmetic._

class PECtrl extends Bundle {
    val mac = Bool() // 乘累加使能
    val acc_ui = Bool() // 累加器方向: 上/下
    val load_reg_li = Bool() // 从左输入加载寄存器
    val load_reg_ui = Bool() // 从上输入加载寄存器
    // pass through
    val flow_lr = Bool() // left -> right
    val flow_ud = Bool() // right -> down
    val flow_du = Bool() // down -> up
    // mac_out -> reg
    val update_reg = Bool() // 用MAC结果更新寄存器
    // compute 2^reg
    val exp2 = Bool() // 指数运算模式

    // getElements might be dangerous, define them manually
    def getCtrlElements: Seq[Bool] = Seq(
      mac,
      acc_ui,
      load_reg_li,
      load_reg_ui,
      flow_lr,
      flow_ud,
      flow_du,
      update_reg,
      exp2
    )
}

@instantiable
class PE[E <: Data: Arithmetic, A <: Data: Arithmetic](ev: ArithmeticImpl[E, A])
    extends Module {
    val (accType, elemType, macGen) = (ev.accType, ev.elemType, ev.peMac _)
    @public val io = IO(new Bundle {
        val in_ctrl = Flipped(Valid(new PECtrl)) // 控制输入
        val out_ctrl = Valid(new PECtrl) // 控制输出

        val u_input = Flipped(Valid(accType)) // 上输入
        val u_output = Valid(accType) // 上输出
        val d_input = Flipped(Valid(accType)) // 下输入
        val d_output = Valid(accType) // 下输出
        val l_input = Flipped(Valid(elemType)) // 左输入
        val r_output = Valid(elemType) // 右输出
    })
    // 乘累加/指数单元
    val macUnit = Module(macGen())
    // PE本地寄存器(元素精度)
    val reg = Reg(elemType)
    val ctrl = io.in_ctrl.bits
    // TODO: this is actually useless, ctrl signals does not depend on fire
    val fire = io.in_ctrl.fire

    // 跟踪exp2运算完成状态
    // as long as exp2 is not the first operation, exp2Done does not need to be reset
    val exp2Done = Reg(Bool())
    when(fire) {
        when(ctrl.exp2) {
            exp2Done := exp2Done || macUnit.io.out_exp2 // 累积完成标志
        }.otherwise({
            exp2Done := false.B // 非exp2模式时重置
        })
    }

    when(fire) { // 当控制信号有效且准备好时
        when(ctrl.load_reg_li) {
            reg := io.l_input.bits // 从左输入加载
        }.elsewhen(ctrl.load_reg_ui) {
            reg := ev.viewAasE(io.u_input.bits) // 从上输入加载(精度转换)
        }.elsewhen(ctrl.update_reg || (macUnit.io.out_exp2 && !exp2Done)) {
            reg := macUnit.io.out_elemType // 用MAC结果更新
        }
    }

    macUnit.io.in_a := reg // A输入:本地寄存器
    macUnit.io.in_b := io.l_input.bits // B输入:左输入
    // C输入选择: 上输入或下输入
    macUnit.io.in_c := Mux(ctrl.acc_ui, io.u_input.bits, io.d_input.bits)
    // 操作模式选择: 指数或乘累加
    macUnit.io.in_cmd := Mux(ctrl.exp2, MacCMD.EXP2, MacCMD.MAC)

    io.out_ctrl := io.in_ctrl

    io.r_output.bits := Mux(ctrl.load_reg_li, reg, io.l_input.bits)
    io.r_output.valid := fire && (ctrl.load_reg_li || ctrl.flow_lr)

    io.d_output.bits := Mux(
      ctrl.mac && ctrl.acc_ui,
      macUnit.io.out_accType,
      io.u_input.bits
    )
    io.d_output.valid := fire && (ctrl.mac && ctrl.acc_ui || ctrl.flow_ud)

    io.u_output.bits := Mux(
      ctrl.mac && !ctrl.acc_ui,
      macUnit.io.out_accType,
      io.d_input.bits
    )
    io.u_output.valid := fire && (ctrl.mac && !ctrl.acc_ui || ctrl.flow_du)
}
