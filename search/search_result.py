"""
赛马娘AI训练框架 - 搜索结果

参考 UmaAi 的 Search.h 中的 SearchResult 结构。
记录每个动作的搜索结果和分数分布。
"""

import math
from typing import Optional
from config import MAX_SCORE, NORM_DISTRIBUTION_SAMPLING


class ModelOutputValue:
    """模型输出值（平均分、标准差、加权分）"""
    
    # 不合法动作的默认值
    ILLEGAL = None  # 延迟初始化

    def __init__(self, score_mean: float = 0.0, score_stdev: float = 0.0, value: float = 0.0):
        self.score_mean = score_mean  # 预测平均分
        self.score_stdev = score_stdev  # 预测标准差
        self.value = value  # 考虑激进度后的加权分
    
    @classmethod
    def _get_illegal(cls):
        if cls.ILLEGAL is None:
            cls.ILLEGAL = ModelOutputValue(1e-5, 0, 1e-5)
        return cls.ILLEGAL


class SearchResult:
    """
    搜索结果
    
    记录某个动作的蒙特卡洛搜索结果。
    包含分数分布、搜索次数等信息。
    """
    
    # 正态分布CDF反函数查找表
    _cdf_inv_table: Optional[list] = None
    
    def __init__(self):
        self.is_legal: bool = False
        self.num: int = 0  # 搜索次数
        # Welford在线统计量（替代final_score_distribution，P1-6/P1-7修复）
        self._count: int = 0   # 总样本量
        self._mean: float = 0.0  # 在线均值
        self._m2: float = 0.0    # 二阶中心矩
        self._min: float = float('inf')   # 最低分
        self._max: float = float('-inf')  # 最高分
        self._up_to_date: bool = True
        self._last_calculate: ModelOutputValue = ModelOutputValue._get_illegal()
        self._last_radical_factor: float = 0.0
    
    @classmethod
    def init_cdf_table(cls):
        """初始化正态分布CDF反函数表"""
        if cls._cdf_inv_table is not None:
            return
        cls._cdf_inv_table = []
        for i in range(NORM_DISTRIBUTION_SAMPLING):
            x = (i + 0.5) / NORM_DISTRIBUTION_SAMPLING
            cls._cdf_inv_table.append(cls._normal_cdf_inverse(x))
    
    @staticmethod
    def _normal_cdf_inverse(p: float) -> float:
        """正态分布CDF的反函数（Rational approximation）"""
        # Peter Acklam的近似算法
        # https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
        a = [-3.969683028665376e+01, 2.209460983245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798588866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758275053992e+00, -2.549732539343734e+00,
              4.374664141464968e+00, 2.938163989965493e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661904975e+00]

        p_low = 0.02425
        p_high = 1 - p_low

        if p < p_low:
            q = math.sqrt(-2 * math.log(p))
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        return x

    def clear(self):
        """清空搜索结果（不重置is_legal，P1-12修复）"""
        self.num = 0
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float('inf')
        self._max = float('-inf')
        self._up_to_date = True
        self._last_calculate = ModelOutputValue._get_illegal()
        self._last_radical_factor = 0.0

    def add_result(self, v: ModelOutputValue):
        """添加一个蒙特卡洛结果（Welford批量更新，P1-6/P1-7修复）"""
        self._up_to_date = False
        self.num += 1
        
        # 批量Welford更新：将NORM_DISTRIBUTION_SAMPLING个采样视为一批
        n_new = NORM_DISTRIBUTION_SAMPLING
        mean_new = v.score_mean
        m2_new = n_new * v.score_stdev * v.score_stdev  # M2 = n * σ²
        n_old = self._count
        n_combined = n_old + n_new
        delta = mean_new - self._mean
        self._mean += n_new * delta / n_combined
        self._m2 += m2_new + n_old * n_new * delta * delta / n_combined
        self._count = n_combined
        
        # 更新min/max
        lo = max(0, v.score_mean - 3.5 * max(v.score_stdev, 0))
        hi = min(MAX_SCORE - 1, v.score_mean + 3.5 * max(v.score_stdev, 0))
        if lo < self._min:
            self._min = lo
        if hi > self._max:
            self._max = hi

    def get_weighted_mean_score(self, radical_factor: float) -> ModelOutputValue:
        """计算加权平均分（按排名加权，Welford统计+正态近似，P1-6/P1-7修复）"""
        if self._up_to_date and self._last_radical_factor == radical_factor:
            return self._last_calculate
        
        if not self.is_legal:
            self._last_calculate = ModelOutputValue._get_illegal()
            return self._last_calculate

        if self._count <= 0:
            self._last_calculate = ModelOutputValue._get_illegal()
            return self._last_calculate

        # Welford统计量直接算mean和stdev
        score_mean = self._mean
        score_stdev = math.sqrt(self._m2 / self._count) if self._count > 0 else 0

        # 用正态近似计算rank加权value
        SearchResult.init_cdf_table()
        total_n_inv = 1.0 / NORM_DISTRIBUTION_SAMPLING
        value_weight_total = 0.0
        value_total = 0.0
        for i in range(NORM_DISTRIBUTION_SAMPLING):
            r = (i + 0.5) * total_n_inv  # 排名比例
            y = score_mean + score_stdev * SearchResult._cdf_inv_table[i]
            w = r ** radical_factor
            value_weight_total += w
            value_total += w * y

        if value_weight_total <= 0:
            self._last_calculate = ModelOutputValue._get_illegal()
            return self._last_calculate

        v = ModelOutputValue(
            score_mean=score_mean,
            score_stdev=score_stdev,
            value=value_total / value_weight_total
        )
        
        self._up_to_date = True
        self._last_radical_factor = radical_factor
        self._last_calculate = v
        return v
