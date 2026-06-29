"""
赛马娘AI训练框架 - 评分模块

从game.py提取的评分计算函数。
包含技能分、最终总分、评价点等。
"""

from config import (
    SCORING_NORMAL,
    # 新增命名常量
    SCORE_ABOVE_100_MULT, SCORING_WEIGHTS, PROPERTY_HALVE_THRESHOLD,
)


def calc_skill_score(game) -> float:
    """技能分"""
    rate = game.pt_score_rate * SCORE_ABOVE_100_MULT if game.is_qie_zhe else game.pt_score_rate
    return rate * game.skill_pt + game.skill_score


def calc_final_score(game) -> int:
    """最终总分"""
    if game.scoring_mode == SCORING_NORMAL:
        return _calc_final_score_rank(game)
    return _calc_final_score_rank(game)  # 默认用评价点模式


def _calc_final_score_rank(game) -> int:
    """评价点计算
    
    属性>100部分的倍率（来源：评价点计算公式）
    """
    total = 0
    for i in range(5):
        stat = min(game.five_status[i], game.five_status_limit[i])
        # 简化的评分函数
        if stat <= 100:
            total += stat
        else:
            total += int(100 + (stat - 100) * SCORE_ABOVE_100_MULT)
    total += int(game.get_skill_score())
    return total


def _calc_final_score_sum(game) -> int:
    """属性之和评分
    
    属性权重 [速,耐,力,根,智]（用于_final_score_sum）
    属性上限超过1200后的折半系数（来源：游戏机制，>1200后每2点算1点）
    """
    total = 0
    for i in range(5):
        real_stat = min(game.five_status[i], game.five_status_limit[i])
        if real_stat > PROPERTY_HALVE_THRESHOLD:
            real_stat = PROPERTY_HALVE_THRESHOLD + (real_stat - PROPERTY_HALVE_THRESHOLD) / 2
        total += SCORING_WEIGHTS[i] * real_stat
    total += game.get_skill_score()
    return int(max(0, total))
