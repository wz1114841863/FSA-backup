package fsa.sa

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy._
import fsa.arithmetic.ArithmeticSyntax._
import fsa.arithmetic.{Arithmetic, ArithmeticImpl}

class SystolicArray[E <: Data: Arithmetic, A <: Data: Arithmetic](
    rows: Int,
    cols: Int
)(implicit ev: ArithmeticImpl[E, A])
    extends Module {

    val io = IO(new Bundle {
        val cmp_ctrl = Flipped(Valid(new CmpControl)) // CMP控制输入
        val pe_ctrl = Flipped(Vec(rows, Valid(new PECtrl))) // 每行PE的控制
        val pe_data = Input(Vec(rows, ev.elemType)) // 每行的输入数据(元素精度)
        val acc_out = Output(Vec(cols, Valid(ev.accType))) // 每列的累加器输出
    })

    /*
      CMP[0]  -> CMP[1]  -> ... -> CMP[col-1]              ← 顶部比较器链
       |          |                 |
      PE[0,0] -> PE[0,1] -> ... -> PE[0,col-1]             ← 第0行PE
       |                            |
      ...                          ...                     ← 中间行PE
       |                            |
      PE[row-1,0] -> ...        -> PE[row-1, col-1]        ← 第row-1行PE
     */

    // 比较器阵列:每列顶部一个CMP
    val cmp_array = Seq.fill(cols) { Module(new CMP(ev)) }
    //  val mesh = Seq.fill(rows) { Seq.fill(cols) { Module(new PE(ev.elemType, ev.accType, ev.peMac _) ) } }

    // PE阵列:使用Chisel Hierarchy API优化
    val peDef = Definition(new PE(ev)) // 定义PE模板
    val mesh = Seq.fill(rows) { Seq.fill(cols) { Instance(peDef) } } // 实例化PE网格
    val meshT = mesh.transpose // 转置,便于按列访问

    def pipe_no_reset[T <: Data](in: Valid[T]) = {
        // 无复位的流水线寄存器
        withReset(false.B) { Pipe(in) }
    }

    // left -> right
    cmp_array.foldLeft(io.cmp_ctrl) { (ctrl, cmp) =>
        cmp.io.in_ctrl := ctrl // 控制信号传递给CMP
        // cmp unit is stateful, need explicit reset
        Pipe(cmp.io.out_ctrl) // 输出控制经过流水线寄存器
    }
    for ((row, in_ctrl) <- mesh.zip(io.pe_ctrl)) {
        row.foldLeft(in_ctrl) { (ctrl, pe) =>
            {
                pe.io.in_ctrl := ctrl
                // 无复位流水线
                pipe_no_reset(pe.io.out_ctrl)
            }
        }
    }
    io.pe_data
        .map(d => { // 将数据包装为Valid接口
            val v = Wire(Valid(ev.elemType))
            v.valid := true.B
            v.bits := d
            v
        })
        .zip(mesh)
        .foreach { case (in_data, row) =>
            row.foldLeft(in_data) { (in, pe) =>
                {
                    // 数据传递给PE的左输入
                    pe.io.l_input := in
                    // 右输出经过流水线
                    pipe_no_reset(pe.io.r_output)
                }
            }
        }

    // up <-> down
    for ((col, cmp) <- meshT.zip(cmp_array)) {
        val cmp_out = pipe_no_reset(cmp.io.d_output) // CMP输出
        // up -> down, CMP输出向下传播到PE列
        col.foldLeft(cmp_out) { (in, pe) =>
            {
                // PE的上输入
                pe.io.u_input := in
                // PE的下输出
                pipe_no_reset(pe.io.d_output)
            }
        }
        // down -> up
        val bottom_in = Wire(Valid(ev.accType))
        // TODO: control the bottom input
        bottom_in.valid := true.B
        bottom_in.bits := ev.accType.zero
        col.reverse.foldLeft(bottom_in) { (in, pe) =>
            {
                // PE的下输入
                pe.io.d_input := in
                // PE的上输出
                pipe_no_reset(pe.io.u_output)
            }
        }
        // 最顶部PE的上输出
        val cmp_in = pipe_no_reset(col.head.io.u_output)
        // 作为CMP的输入
        cmp.io.d_input := cmp_in
    }

    for ((io_out, pe) <- io.acc_out.zip(meshT.map(_.last))) {
        io_out := pipe_no_reset(pe.io.d_output)
    }

}
