"""
赛马娘AI训练框架 - 评分模块

从game.py提取的评分计算函数。
包含技能分、最终总分、评价点等。
修复P0-8：用整数算术替代float除法，Race/Mile模式不再静默fallback到rank模式。
"""

from config import (
    SCORING_NORMAL, SCORING_RACE, SCORING_MILE,
    # 命名常量
    SCORE_ABOVE_100_MULT, SCORING_WEIGHTS, PROPERTY_HALVE_THRESHOLD,
)


def calc_skill_score(game) -> float:
    """技能分"""
    rate = game.pt_score_rate * SCORE_ABOVE_100_MULT if game.is_qie_zhe else game.pt_score_rate
    return rate * game.skill_pt + game.skill_score


def calc_final_score(game) -> int:
    """最终总分
    
    根据评分模式分发到对应计算函数：
    - NORMAL(0): 评价点模式
    - RACE(1): 通用大赛模式
    - MILE(2): 英里模式
    """
    if game.scoring_mode == SCORING_NORMAL:
        return _calc_final_score_rank(game)
    elif game.scoring_mode == SCORING_RACE:
        return _calc_final_score_race(game)
    elif game.scoring_mode == SCORING_MILE:
        return _calc_final_score_mile(game)
    else:
        # 未知模式回退到评价点
        return _calc_final_score_rank(game)


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
    # 剧本额外评分（如Ramen的隠し味评分）
    if game.scenario is not None:
        total += game.scenario.calculate_score(game)
    return total


def _calc_final_score_sum(game) -> int:
    """属性之和评分
    
    属性权重 [速,耐,力,根,智]（用于_final_score_sum）
    属性上限超过1200后的折半系数（来源：游戏机制，>1200后每2点算1点）
    
    修复P0-8：使用//整数除法替代/浮点除法，避免精度问题
    """
    total = 0
    for i in range(5):
        real_stat = min(game.five_status[i], game.five_status_limit[i])
        if real_stat > PROPERTY_HALVE_THRESHOLD:
            # 修复：用//2整数除法替代/2浮点除法
            real_stat = PROPERTY_HALVE_THRESHOLD + (real_stat - PROPERTY_HALVE_THRESHOLD) // 2
        total += SCORING_WEIGHTS[i] * real_stat
    total += game.get_skill_score()
    # 剧本额外评分
    if game.scenario is not None:
        total += game.scenario.calculate_score(game)
    return int(max(0, total))


def _calc_final_score_race(game) -> int:
    """通用大赛模式评分
    
    大赛模式使用属性加权和 + 技能分，属性>1200部分折半。
    与Mile模式的区别：权重侧重速度和耐力。
    
    TODO: 需确认CY大赛评分的具体权重公式，
    当前使用SCORING_WEIGHTS作为近似，与_sum模式相同。
    待获取官方评分规则后修正权重。
    """
    total = 0
    for i in range(5):
        real_stat = min(game.five_status[i], game.five_status_limit[i])
        if real_stat > PROPERTY_HALVE_THRESHOLD:
            real_stat = PROPERTY_HALVE_THRESHOLD + (real_stat - PROPERTY_HALVE_THRESHOLD) // 2
        total += SCORING_WEIGHTS[i] * real_stat
    total += game.get_skill_score()
    # 剧本额外评分
    if game.scenario is not None:
        total += game.scenario.calculate_score(game)
    return int(max(0, total))


def _calc_final_score_mile(game) -> int:
    """英里模式评分
    
    英里模式使用属性加权和 + 技能分，属性>1200部分折半。
    与Race模式的区别：权重侧重速度和力量。
    
    TODO: 需确认CY英里赛评分的具体权重公式，
    当前使用SCORING_WEIGHTS作为近似，与_sum模式相同。
    待获取官方评分规则后修正权重。
    """
    total = 0
    for i in range(5):
        real_stat = min(game.five_status[i], game.five_status_limit[i])
        if real_stat > PROPERTY_HALVE_THRESHOLD:
            real_stat = PROPERTY_HALVE_THRESHOLD + (real_stat - PROPERTY_HALVE_THRESHOLD) // 2
        total += SCORING_WEIGHTS[i] * real_stat
    total += game.get_skill_score()
    # 剧本额外评分
    if game.scenario is not None:
        total += game.scenario.calculate_score(game)
    return int(max(0, total))
