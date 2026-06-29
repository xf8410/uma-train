"""
赛马娘AI训练框架 - NN输入编码

参考 UmaAi 的 NNInput.h 和 Game::getNNInputV1，将游戏状态编码为神经网络输入向量。
"""

import math
from typing import List
from simulator.game import Game
from simulator.action import TrainActionType
from simulator.person import PersonType, FriendStage
from simulator.bad_condition import BadConditionType
from simulator.scenarios.ramen import RamenScenario
from config import (
    NN_INPUT_C, NN_INPUT_C_GLOBAL, NN_INPUT_C_CARD, NN_INPUT_C_PERSON,
    NN_INPUT_C_CARDPERSON, NN_CARD_NUM, TOTAL_TURN,
    NN_INPUT_C_BC, NN_INPUT_C_RAMEN,
)


def encode_game_state(game: Game) -> List[float]:
    """将游戏状态编码为神经网络输入向量
    
    输入结构：
    (全局信息587维)(支援卡1信息89+12维)...(支援卡6信息89+12维)
    
    参考 UmaAi 的 Game::getNNInputV1
    
    Args:
        game: 游戏状态
        
    Returns:
        长度为 NN_INPUT_C 的浮点数列表
    """
    buf = [0.0] * NN_INPUT_C
    idx = 0
    
    # ===== 全局信息 (587维) =====
    # 回合数（归一化）
    buf[idx] = game.turn / TOTAL_TURN; idx += 1
    
    # 体力
    buf[idx] = game.vital / 120.0; idx += 1
    buf[idx] = game.max_vital / 120.0; idx += 1
    
    # 干劲 (one-hot 5维)
    for i in range(5):
        buf[idx] = 1.0 if game.motivation == i + 1 else 0.0; idx += 1
    
    # 五维属性（归一化到0-1）
    for i in range(5):
        buf[idx] = game.five_status[i] / 2000.0; idx += 1
    # 五维上限
    for i in range(5):
        buf[idx] = game.five_status_limit[i] / 2000.0; idx += 1
    # 成长率
    for i in range(5):
        buf[idx] = game.five_status_bonus[i] / 120.0; idx += 1
    
    # 技能点
    buf[idx] = game.skill_pt / 1000.0; idx += 1
    # 技能分
    buf[idx] = game.skill_score / 5000.0; idx += 1
    
    # 训练等级
    for i in range(5):
        buf[idx] = game.get_training_level(i) / 4.0; idx += 1
    
    # 训练值 (5*6=30维)
    for t in range(5):
        for i in range(6):
            buf[idx] = game.train_value[t][i] / 100.0; idx += 1
    
    # 体力变化
    for i in range(5):
        buf[idx] = game.train_vital_change[i] / 50.0; idx += 1
    
    # 失败率
    for i in range(5):
        buf[idx] = game.fail_rate[i] / 100.0; idx += 1
    
    # 是否闪彩
    for i in range(5):
        buf[idx] = 1.0 if game.is_train_shining[i] else 0.0; idx += 1
    
    # 是否比赛回合
    buf[idx] = 1.0 if game.is_racing else 0.0; idx += 1
    
    # 是否夏合宿
    buf[idx] = 1.0 if game.is_xiahesu() else 0.0; idx += 1
    
    # 是否可比赛
    buf[idx] = 1.0 if game.is_race_available() else 0.0; idx += 1
    
    # 失败率bias
    buf[idx] = game.failure_rate_bias / 4.0; idx += 1
    
    # 特殊状态
    buf[idx] = 1.0 if game.is_qie_zhe else 0.0; idx += 1
    buf[idx] = 1.0 if game.is_ai_jiao else 0.0; idx += 1
    buf[idx] = 1.0 if game.is_positive_thinking else 0.0; idx += 1
    buf[idx] = 1.0 if game.is_refresh_mind else 0.0; idx += 1
    
    # 友人卡状态
    buf[idx] = 1.0 if game.friend_type != 0 else 0.0; idx += 1
    buf[idx] = 1.0 if game.friend_is_ssr else 0.0; idx += 1
    buf[idx] = game.friend_stage / 3.0; idx += 1
    buf[idx] = game.friend_outgoing_used / 5.0; idx += 1
    
    # 非卡羁绊
    buf[idx] = game.friendship_noncard_yayoi / 100.0; idx += 1
    buf[idx] = game.friendship_noncard_reporter / 100.0; idx += 1
    
    # 种马蓝因子
    for i in range(5):
        buf[idx] = game.zhong_ma_blue_count[i] / 6.0; idx += 1
    
    # Dreams剧本状态
    buf[idx] = game.mecha_en / 10.0; idx += 1
    buf[idx] = game.mecha_overdrive_energy / 6.0; idx += 1
    buf[idx] = 1.0 if game.mecha_overdrive_enabled else 0.0; idx += 1
    buf[idx] = 1.0 if game.mecha_any_lose else 0.0; idx += 1
    
    # 研究等级
    for i in range(5):
        buf[idx] = game.mecha_rival_lv[i] / 700.0; idx += 1
    buf[idx] = game.mecha_rival_lv_total / 3500.0; idx += 1
    
    # 机甲升级
    for i in range(3):
        for j in range(3):
            buf[idx] = game.mecha_upgrade[i][j] / 5.0; idx += 1
    for i in range(3):
        buf[idx] = game.mecha_upgrade_total[i] / 15.0; idx += 1
    
    # 齿轮状态
    for i in range(5):
        buf[idx] = 1.0 if game.mecha_has_gear[i] else 0.0; idx += 1
    
    # UGE胜负
    for i in range(5):
        buf[idx] = game.mecha_win_history[i] / 2.0; idx += 1
    
    # 训练倍率
    for i in range(6):
        buf[idx] = game.mecha_training_status_multiplier[i] - 1.0; idx += 1
    
    # ===== バッドコンディション状态 (8维) =====
    # 6种状态one-hot
    for bc_type in [BadConditionType.BAD, BadConditionType.LAZY,
                    BadConditionType.FAT, BadConditionType.HEADACHE,
                    BadConditionType.SKIN, BadConditionType.LATE_BED]:
        buf[idx] = 1.0 if game.bc_manager.has(bc_type) else 0.0; idx += 1
    # BC数量（归一化）
    buf[idx] = game.bc_manager.count / 6.0; idx += 1
    # 是否可治愈（保健室/お休み）
    has_healable = (game.bc_manager.count > 0)
    buf[idx] = 1.0 if has_healable else 0.0; idx += 1

    # ===== Ramen剧本状态 (15维) =====
    ramen = getattr(game, '_ramen_scenario', None)
    if ramen is not None:
        # 隠し味の秘訣数量（归一化到~50）
        buf[idx] = ramen.kakushimi_count / 50.0; idx += 1
        # 試食会次数
        buf[idx] = ramen.tasting_count / 5.0; idx += 1
        # コツ等级（5维）
        for i in range(5):
            buf[idx] = ramen.kotsu_level[i] / 3.0; idx += 1
        # CheckPoint进度
        if ramen.expected_checkpoint_pt > 0:
            buf[idx] = ramen.checkpoint_pt / ramen.expected_checkpoint_pt; idx += 1
        else:
            buf[idx] = 0.0; idx += 1
        # 地域贡献（5个区域，归一化）
        for region_id in range(5):
            buf[idx] = ramen.region_points.get(region_id, 0) / 100.0; idx += 1
        # SpecialFeelingNum
        buf[idx] = ramen.special_feeling_num / 10.0; idx += 1
        # Feeling值（2维：取前2个feeling_type的值）
        feeling_vals = list(ramen.feeling_values.values())
        buf[idx] = feeling_vals[0] / 100.0 if len(feeling_vals) > 0 else 0.0; idx += 1
        buf[idx] = feeling_vals[1] / 100.0 if len(feeling_vals) > 1 else 0.0; idx += 1
    else:
        # 非Ramen剧本：15维全0
        for _ in range(NN_INPUT_C_RAMEN):
            buf[idx] = 0.0; idx += 1

    # 剩余全局维度用0填充（到NN_INPUT_C_GLOBAL维）
    # idx目前约132+8+15=155，需要填充到587
    
    # ===== 支援卡信息 =====
    for card_idx in range(NN_CARD_NUM):
        card_start = NN_INPUT_C_GLOBAL + card_idx * NN_INPUT_C_CARDPERSON
        
        if card_idx < len(game.persons):
            p = game.persons[card_idx]
            cp = p.card_param
            
            # 支援卡参数（NN_INPUT_C_CARD=89维）
            ci = card_start
            buf[ci] = cp.card_type / 6.0; ci += 1
            buf[ci] = cp.you_qing_basic / 100.0; ci += 1
            buf[ci] = cp.gan_jing_basic / 100.0; ci += 1
            buf[ci] = cp.xun_lian_basic / 100.0; ci += 1
            for i in range(6):
                buf[ci] = cp.bonus_basic[i] / 50.0; ci += 1
            buf[ci] = cp.wiz_vital_bonus / 10.0; ci += 1
            for i in range(6):
                buf[ci] = cp.initial_bonus[i] / 30.0; ci += 1
            buf[ci] = cp.hint_level / 5.0; ci += 1
            buf[ci] = cp.hint_prob_increase / 100.0; ci += 1
            buf[ci] = cp.de_yi_lv / 100.0; ci += 1
            buf[ci] = cp.fail_rate_drop / 100.0; ci += 1
            buf[ci] = cp.vital_cost_drop / 100.0; ci += 1
            buf[ci] = 1.0 if cp.is_link else 0.0; ci += 1
            buf[ci] = cp.sai_hou / 50.0; ci += 1
            buf[ci] = cp.event_recovery_amount_up / 50.0; ci += 1
            buf[ci] = cp.event_effect_up / 50.0; ci += 1
            # 剩余卡片参数用0填充到89维
            
            # 人头信息（NN_INPUT_C_PERSON=12维）
            pi = card_start + NN_INPUT_C_CARD
            buf[pi] = p.friendship / 100.0; pi += 1   # 1
            buf[pi] = 1.0 if p.is_hint else 0.0; pi += 1  # 2
            buf[pi] = p.person_type / 7.0; pi += 1  # 3
            buf[pi] = p.card_record / 10.0; pi += 1  # 4
            
            # 该人头在哪个训练（one-hot 5维）
            for t in range(5):
                found = any(
                    game.person_distribution[t][h] == card_idx
                    for h in range(5)
                )
                buf[pi] = 1.0 if found else 0.0; pi += 1  # 5-9
            
            # 闪彩状态 (3维 - 对应卡片类型的训练 + 智力训练)
            buf[pi] = 1.0 if p.is_card_shining(cp.card_type) else 0.0; pi += 1  # 10
            buf[pi] = 1.0 if p.is_card_shining(4) else 0.0; pi += 1  # 11
            
            # 是否是友人卡
            buf[pi] = 1.0 if p.person_type == PersonType.FRIEND_CARD else 0.0; pi += 1  # 12
    
    return buf
