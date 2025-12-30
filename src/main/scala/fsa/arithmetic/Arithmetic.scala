package fsa.arithmetic

import chisel3._

trait Arithmetic[T] {
    // 为类型 T 自动添加算术操作方法, 将类型T转换为对应的算术操作对象
    // 进而让任意类型 T 可以"拥有"算术运算的能力.
    implicit def cast(self: T): ArithmeticOps[T]
}

trait ArithmeticOps[T] {
    def zero: T // 零值
    def one: T // 一值
    def minimum: T // 最小值
    // log2(e) / sqrt(dk)
    def attentionScale(dk: Int): T // 注意力缩放因子
    def typeRepr: String //
}

object Arithmetic {
    // 为FloatPoint类型提供Arithmetic实例
    implicit object FPArithmetic extends Arithmetic[FloatPoint] {
        // 使用easyfloat库实现浮点数的算术操作
        import easyfloat.{IEEEFloat, PyFPConst}
        override implicit def cast(
            self: FloatPoint
        ): ArithmeticOps[FloatPoint] = new ArithmeticOps[FloatPoint] {
            override def zero = 0.U.asTypeOf(self)
            override def one = {
                val bits =
                    IEEEFloat.expBias(self.expWidth) << self.mantissaWidth
                bits.U.asTypeOf(self)
            }
            override def minimum = {
                // -inf
                val sign = BigInt(1) << (self.expWidth + self.mantissaWidth)
                val exp =
                    ((BigInt(1) << self.expWidth) - 1) << self.mantissaWidth
                val bits = sign | exp
                bits.U.asTypeOf(self)
            }
            override def attentionScale(dk: Int) = PyFPConst
                .attentionScale(
                  self.expWidth,
                  self.mantissaWidth,
                  dk = dk,
                  projectDir = "generators/easyfloat"
                )
                .U
                .asTypeOf(self)

            override def typeRepr: String =
                (self.expWidth, self.mantissaWidth) match {
                    case (5, 10) => "fp16"
                    case (8, 7)  => "bf16"
                    case (8, 23) => "fp32"
                    case (e, m)  => f"e${e}m$m"
                }
        }
    }
}

// 语法糖支持
// 通过隐式转换, 允许直接调用data.zero而不是arith.cast(data).zero
// import this to directly access T.zero / T.minimum
object ArithmeticSyntax {
    implicit class ArithmeticOpsSyntax[T](self: T)(implicit
        arith: Arithmetic[T]
    ) {
        import arith._
        def zero: T = self.zero
        def one: T = self.one
        def minimum: T = self.minimum
        def attentionScale(dk: Int): T = self.attentionScale(dk)
        def typeRepr: String = self.typeRepr
    }
}

// 乘累加指令
object MacCMD {
    def width = 1
    def MAC = 0.U(width.W) // 乘累加
    def EXP2 = 1.U(width.W) // 指数运算(分段线性近似)
}

// 比较命令
object CmpCMD {
    def width = 1
    def MAX = 0.U(width.W) // 取最大值
    def SUB = 1.U(width.W) // 求差值
}

// 乘累加单元
abstract class MacUnit[E <: Data: Arithmetic, A <: Data: Arithmetic](
    val elemType: E,
    val accType: A
) extends Module {
    val io = IO(new Bundle {
        val in_a = Input(elemType) // reg in PE
        val in_b = Input(elemType) // left input
        val in_c = Input(accType) // up/down input
        val in_cmd = Input(UInt(MacCMD.width.W)) // 操作命令
        val out_accType = Output(accType) // 累加器类型输出
        val out_elemType = Output(elemType) // 元素类型输出
        val out_exp2 = Output(Bool()) // exp2完成标志
    })
}

// 比较单元
abstract class CmpUnit[A <: Data](val accType: A) extends Module {
    val io = IO(new Bundle {
        val in_a = Input(accType)
        val in_b = Input(accType)
        val out_max = Output(accType) // 最大值
        val out_diff = Output(accType) // 差值
    })
}

trait HasMultiCycleIO { this: Module =>
    val multiCycleIO = IO(new Bundle {
        val reciprocal_in_valid = Input(Bool()) // 倒数运算输入有效
        val reciprocal_out_valid = Output(Bool()) // 倒数运算输出有效
    })
}

trait HasArithmeticParams {
    val reciprocalLatency: Int // 倒数运算延迟周期数
    val exp2PwlPieces: Int // exp2分段线性近似的段数
}

abstract class ArithmeticImpl[E <: Data: Arithmetic, A <: Data: Arithmetic]
    extends HasArithmeticParams {
    def elemType: E // 元素类型
    def accType: A // 累加器类型

    def peMac: MacUnit[E, A] // PE中的乘累加单元
    def accUnit: MacUnit[A, A] with HasMultiCycleIO // 累加器单元
    def accCmp: CmpUnit[A] // 累加器比较单元
    
    def exp2PwlIntercepts: Seq[A] // exp2分段线性近似的截距
    def exp2PwlSlopes: Seq[E] // exp2分段线性斜率表

    def viewAasE: A => E // 将累加器类型看作元素类型(类型转换)
    def viewEasA: E => A // 将元素类型看作累加器类型
    def cvtAtoE: A => E // 累加器类型到元素类型的转换
}
