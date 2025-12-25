import numpy as np
from pyeasyfloat.float import FloatPoint
from pyeasyfloat.rounding import round_raw_float
from pyeasyfloat.backend import BaseFPBackend, PyEasyFloatBackend

type Matrix = list[list[FloatPoint]]


def mat_hex_str(mat: Matrix) -> str:
    """将浮点数矩阵转换为十六进制表示字符串"""
    s = ""
    for row in mat:
        for e in row:
            l = (1 + e.ew + e.mw + 1) // 4
            # 转换为固定长度的十六进制
            s += format(e.to_bits(), f"0{l}x")
            s += " "
        s += "\n"
    return s


def mat_to_numpy_array(mat: Matrix) -> np.ndarray:
    """将自定义浮点数矩阵转换为NumPy数组"""
    return np.array([[x.to_numpy() for x in row] for row in mat])


def np_to_fp(
    x: np.float16 | np.float32 | np.float64 | float, ew: int, mw: int
) -> FloatPoint:
    """将NumPy浮点数转换为指定格式的 FloatPoint"""
    if isinstance(x, float):
        x = np.float64(x)
    fp = FloatPoint.from_numpy(x)
    # 检查位宽是否匹配目标格式 (ew, mw)
    np_ew, np_mw = fp.ew, fp.mw
    if (np_ew, np_mw) != (ew, mw):
        fp = round_raw_float(fp.to_raw(), ew, mw)
    return fp


def fp_to_np(x: FloatPoint) -> np.float64 | np.float32 | np.float16:
    """将 FloatPoint 转换回NumPy浮点数"""
    return x.to_numpy()


def build_mat_from_numpy(arr: np.ndarray, ew: int, mw: int) -> Matrix:
    """从NumPy数组构建自定义浮点数矩阵s"""
    return [[np_to_fp(x, ew, mw) for x in row] for row in arr]


def neg_fp(x: FloatPoint) -> FloatPoint:
    """对 FloatPoint 取负(改变符号位)"""
    nx = FloatPoint(x.ew, x.mw)
    nx.sign = not x.sign
    nx.exp = x.exp
    nx.mantissa = x.mantissa
    return nx


class FlashAttentionTile:
    """FlashAttention

    输入: Q, K, V矩阵 + 前一个块的累加器
           ↓
    1. 计算 S = Q x Kᵀ / √d (注意力分数)
           ↓
    2. 在线softmax计算(处理数值稳定性)
           ↓
    3. 计算 O = softmax(S) x V (输出)
           ↓
    4. 更新全局累加器
    """

    backend: BaseFPBackend

    # 输入矩阵
    Q: Matrix  # [Br, d], 查询块
    K: Matrix  # [Bc, d], 键块
    V: Matrix  # [Bc, d], 值块

    # 中间计算结果(累加精度)
    S: Matrix  # [Br, Bc] - 注意力分数 Q×Kᵀ
    S_low_precision: Matrix

    # 在线softmax状态
    PrevRowMax: Matrix  # [Br, 1] - 前一个块的行最大值
    RowMaxS: Matrix  # [Br, 1] - 当前块的行最大值
    AccRowMaxS: Matrix  # [Br, 1] - 累积的行最大值
    NegRowMaxS: Matrix  # [Br, 1] - 取负的行最大值
    DeltaRowMax: Matrix  # [Br, 1] - 行最大值变化量
    ExpDeltaRowMaxS1: Matrix
    ExpDeltaRowMaxS2: Matrix

    SMinusRowMax: Matrix
    SExpStage1: Matrix

    # softmax中间结果
    P: Matrix  # [Br, Bc] - softmax概率
    RowSum: Matrix  # [Br, 1]  - 行求和(归一化分母)

    # 输出相关
    O: Matrix  # [Br, d]  - 当前块输出
    AccRowSum: Matrix
    AccRowSumReciprocal: Matrix
    AccO: Matrix  # [Br, d]  - 累积输出
    NormO: Matrix  # [Br, d]  - 归一化输出

    def __init__(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        PrevRowMax: np.ndarray | Matrix,
        PrevRowSum: np.ndarray | Matrix,
        PrevO: np.ndarray | Matrix,
        mul_ew: int,  # 乘法器精度(如fp16)
        mul_mw: int,
        acc_ew: int,  # 累加器精度(如fp32)
        acc_mw: int,
        backend: BaseFPBackend,
    ):
        self.backend = backend
        self.mul_ew, self.mul_mw = mul_ew, mul_mw  # 计算精度(低)
        self.acc_ew, self.acc_mw = acc_ew, acc_mw  # 累加精度(高)

        # Q, K, V用乘法精度存储
        self.Q = build_mat_from_numpy(Q, mul_ew, mul_mw)
        self.K = build_mat_from_numpy(K, mul_ew, mul_mw)
        self.V = build_mat_from_numpy(V, mul_ew, mul_mw)

        self.PrevRowMax = (
            PrevRowMax
            if isinstance(PrevRowMax, list)
            else build_mat_from_numpy(PrevRowMax, acc_ew, acc_mw)
        )
        self.AccRowSum = (
            PrevRowSum
            if isinstance(PrevRowSum, list)
            else build_mat_from_numpy(PrevRowSum, acc_ew, acc_mw)
        )
        self.AccO = (
            PrevO
            if isinstance(PrevO, list)
            else build_mat_from_numpy(PrevO, acc_ew, acc_mw)
        )

        br, d, bc = len(Q), len(Q[0]), len(K)
        # 中间结果用累加精度存储
        self.S = [
            [FloatPoint.from_bits(0, acc_ew, acc_mw) for _ in range(bc)]
            for _ in range(br)
        ]
        self.O = [
            [FloatPoint.from_bits(0, acc_ew, acc_mw) for _ in range(d)]
            for _ in range(br)
        ]

        self.__mul_qk()

        self.RowMaxS = [
            [self.__max(self.S[row] + self.PrevRowMax[row])] for row in range(br)
        ]
        self.NegRowMaxS = [[neg_fp(x) for x in row] for row in self.RowMaxS]
        self.DeltaRowMax = [
            [self.__sub(self.PrevRowMax[row][0], self.RowMaxS[row][0])]
            for row in range(br)
        ]

        self.AccRowMaxS = [
            [
                (
                    self.RowMaxS[row][0]
                    if self.DeltaRowMax[row][0].sign
                    else self.PrevRowMax[row][0]
                )
            ]
            for row in range(br)
        ]

        log2e_over_sqrt_d = np_to_fp(np.log2(np.e) / np.sqrt(d), acc_ew, acc_mw)
        zero_fp = np_to_fp(np.float32(0), acc_ew, acc_mw)

        self.ExpDeltaRowMaxS1 = [
            [self.backend.fma(row[0], log2e_over_sqrt_d, zero_fp, acc_ew, acc_mw)]
            for row in self.DeltaRowMax
        ]

        self.ExpDeltaRowMaxS2 = [
            [self.backend.exp2(row[0], acc_ew, acc_mw, acc_ew, acc_mw, acc_ew, acc_mw)]
            for row in self.ExpDeltaRowMaxS1
        ]

        self.RowSum = [[np_to_fp(np.float32(0), acc_ew, acc_mw)] for _ in range(br)]

        self.S_low_precision = [
            [round_raw_float(x.to_raw(), mul_ew, mul_mw) for x in row] for row in self.S
        ]

        # 稳定计算:exp(x - max(x))
        # 1. 计算 S - row_max(在低精度下)
        self.SMinusRowMax = [
            [
                self.__sub(self.S_low_precision[row][col], self.RowMaxS[row][0])
                for col in range(bc)
            ]
            for row in range(br)
        ]

        # 2. 计算 log2(e) * (S - row_max) / √d
        mul_log2e = np_to_fp(np.log2(np.e) / np.sqrt(d), mul_ew, mul_mw)
        zero_acc = np_to_fp(np.float32(0), acc_ew, acc_mw)

        self.SExpStage1 = [
            [
                self.backend.fma(
                    self.SMinusRowMax[row][col], mul_log2e, zero_acc, mul_ew, mul_mw
                )
                for col in range(bc)
            ]
            for row in range(br)
        ]

        # 3. 计算 2^{...}(相当于exp)
        # exp(x) = 2^{x * log2(e)}
        self.P = [
            [
                self.backend.exp2(
                    self.SExpStage1[row][col],
                    mul_ew,
                    mul_mw,
                    mul_ew,
                    mul_mw,
                    acc_ew,
                    acc_mw,
                )
                for col in range(bc)
            ]
            for row in range(br)
        ]

        one_fp = np_to_fp(np.float32(1), mul_ew, mul_mw)
        for row in range(br):
            for col in range(bc):
                self.RowSum[row][0] = self.backend.fma(
                    self.P[row][col], one_fp, self.RowSum[row][0], acc_ew, acc_mw
                )

        self.__mul_pv()
        self.__update_global()

    def __sub(self, a: FloatPoint, b: FloatPoint) -> FloatPoint:
        return self.backend.fma(a, np_to_fp(1.0, a.ew, a.mw), neg_fp(b), a.ew, a.mw)

    def __max(self, row: list[FloatPoint]) -> FloatPoint:
        m = row[0]
        for e in row[1:]:
            if self.__sub(m, e).sign:
                m = e
        return m

    def __mul_qk(self):
        # 融合乘加(FMA)操作
        # FMA: fused multiply-add (a * b + c)
        # 硬件通常有FMA单元,比分开乘/加更精确
        br, d = len(self.Q), len(self.Q[0])
        bc = len(self.K)
        for row in range(br):
            for col in range(bc):
                for k in reversed(range(d)):
                    self.S[row][col] = self.backend.fma(
                        self.K[col][k],
                        self.Q[row][k],
                        self.S[row][col],
                        self.S[row][col].ew,
                        self.S[row][col].mw,
                    )

    def __mul_pv(self):
        br, bc = len(self.P), len(self.P[0])
        d = len(self.V[0])
        for row in range(br):
            for col in range(d):
                for i in reversed(range(bc)):
                    self.O[row][col] = self.backend.fma(
                        self.P[row][i],
                        self.V[i][col],
                        self.O[row][col],
                        self.O[row][col].ew,
                        self.O[row][col].mw,
                    )

    def __update_global(self):
        self.AccRowSumReciprocal = []
        self.NormO = []

        for row in range(len(self.RowSum)):
            old_sum = self.AccRowSum[row][0]
            new_sum = self.RowSum[row][0]
            scale = self.ExpDeltaRowMaxS2[row][0]

            self.AccRowSum[row][0] = self.backend.fma(
                old_sum, scale, new_sum, old_sum.ew, old_sum.mw
            )
            reciprocal = self.backend.reciprocal(self.AccRowSum[row][0])
            self.AccRowSumReciprocal.append([reciprocal])

            norm_row = []
            for col in range(len(self.O[0])):
                old_o = self.AccO[row][col]
                new_o = self.O[row][col]
                self.AccO[row][col] = self.backend.fma(
                    old_o, scale, new_o, old_o.ew, old_o.mw
                )
                norm = self.backend.fma(
                    self.AccO[row][col],
                    reciprocal,
                    FloatPoint.from_bits(0, self.acc_ew, self.acc_mw),
                    old_o.ew,
                    old_o.mw,
                )
                norm_row.append(norm)
            self.NormO.append(norm_row)

    def __str__(self) -> str:
        def to_str(name: str, mat: Matrix) -> str:
            return f"{name} hex:\n{mat_hex_str(mat)}{name} float:\n{mat_to_numpy_array(mat)}\n"

        return "\n".join(
            [
                to_str("Q", self.Q),
                to_str("K", self.K),
                to_str("V", self.V),
                to_str("S", self.S),
                to_str("PrevRowMax", self.PrevRowMax),
                to_str("RowMaxS", self.RowMaxS),
                to_str("-RowMaxS", self.NegRowMaxS),
                to_str("DeltaRowMax", self.DeltaRowMax),
                to_str("ExpDeltaRowMaxS1", self.ExpDeltaRowMaxS1),
                to_str("ExpDeltaRowMaxS2", self.ExpDeltaRowMaxS2),
                to_str("SMinusRowMax", self.SMinusRowMax),
                to_str("SExpS1", self.SExpStage1),
                to_str("P", self.P),
                to_str("RowSum", self.RowSum),
                to_str("O", self.O),
                to_str("AccRowSum", self.AccRowSum),
                to_str("AccRowSumReciprocal", self.AccRowSumReciprocal),
                to_str("AccO", self.AccO),
                to_str("NormO", self.NormO),
            ]
        )
