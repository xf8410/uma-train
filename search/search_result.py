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
        self.final_score_distribution: list = [0] * MAX_SCORE  # 分数分布
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
        """清空搜索结果"""
        self.is_legal = False
        self.num = 0
        self.final_score_distribution = [0] * MAX_SCORE
        self._up_to_date = True
        self._last_calculate = ModelOutputValue._get_illegal()
        self._last_radical_factor = 0.0

    def add_result(self, v: ModelOutputValue):
        """添加一个蒙特卡洛结果"""
        self._up_to_date = False
        self.num += 1
        
        # 将正态分布采样到分数分布上
        SearchResult.init_cdf_table()
        for i in range(NORM_DISTRIBUTION_SAMPLING):
            y = int(v.score_mean + v.score_stdev * SearchResult._cdf_inv_table[i] + 0.5)
            y = max(0, min(y, MAX_SCORE - 1))
            self.final_score_distribution[y] += 1

    def get_weighted_mean_score(self, radical_factor: float) -> ModelOutputValue:
        """计算加权平均分（按排名加权）"""
        if self._up_to_date and self._last_radical_factor == radical_factor:
            return self._last_calculate
        
        if not self.is_legal:
            self._last_calculate = ModelOutputValue._get_illegal()
            return self._last_calculate

        n = 0.0  # 总样本量
        score_total = 0.0
        score_sqr_total = 0.0
        value_weight_total = 0.0
        value_total = 0.0
        total_n_inv = 1.0 / (self.num * NORM_DISTRIBUTION_SAMPLING) if self.num > 0 else 0

        for s in range(MAX_SCORE):
            count = self.final_score_distribution[s]
            r = (n + 0.5 * count) * total_n_inv  # 排名比例
            n += count
            score_total += count * s
            score_sqr_total += count * s * s
            
            # 按排名加权
            w = r ** radical_factor
            value_weight_total += w * count
            value_total += w * count * s

        if n <= 0 or value_weight_total <= 0:
            self._last_calculate = ModelOutputValue._get_illegal()
            return self._last_calculate

        v = ModelOutputValue(
            score_mean=score_total / n,
            score_stdev=math.sqrt(max(0, score_sqr_total * n - score_total * score_total)) / n if n > 0 else 0,
            value=value_total / value_weight_total
        )
        
        self._up_to_date = True
        self._last_radical_factor = radical_factor
        self._last_calculate = v
        return v
