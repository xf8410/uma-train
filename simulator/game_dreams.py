"""
赛马娘AI训练框架 - Dreams剧本专属模块

从game.py提取的Dreams/机甲剧本相关函数。
包含UGE比赛、overdrive、EN分配、link角色判断等。
"""

from .action import GameStage
from .person import PersonType
from config import (
    MAX_PERSON_PER_TRAIN, MECHA_TARGET_TOTAL_LEVEL,
    MECHA_GEAR_PROB, MECHA_GEAR_PROB_LINK_BONUS,
    # 新增命名常量
    OVERDRIVE_INVITE_MAX_TRY, OVERDRIVE_EN_COST, OVERDRIVE_TRAIN_MULT,
    URA_START_TURN,
)


def mecha_maybe_run_uge(game) -> bool:
    """检查是否触发UGE比赛"""
    uge_turns = [1, 23, 35, 47, 59, 71]
    if game.turn not in uge_turns:
        return False

    game.game_stage = GameStage.BEFORE_MECHA_UPGRADE
    if game.turn == 1:
        return True

    uge_count = game.turn // 12 - 1
    total_lv = sum(game.mecha_rival_lv)
    target = MECHA_TARGET_TOTAL_LEVEL[uge_count]

    if total_lv >= target:
        # S评价
        game.mecha_en += 6
        game.add_all_status(10 + 5 * uge_count)
        game.skill_pt += 25 + 10 * uge_count
        if uge_count in (0, 2, 4):
            for i in range(5):
                game.add_training_level_count(i, 4)
        game.mecha_win_history[uge_count] = 2
    elif total_lv >= target * 7 // 10:
        # A评价
        game.mecha_en += 5
        game.add_all_status(10 + 5 * uge_count)
        game.skill_pt += 25 + 10 * uge_count
        if uge_count in (0, 2, 4):
            for i in range(5):
                game.add_training_level_count(i, 4)
        game.mecha_win_history[uge_count] = 1
        game.mecha_any_lose = True
    else:
        # B评价
        game.mecha_en += 4
        game.add_all_status(5 + 5 * uge_count)
        game.skill_pt += 20 + 10 * uge_count
        game.mecha_win_history[uge_count] = 0
        game.mecha_any_lose = True

    return True


def mecha_activate_overdrive(game, rng) -> bool:
    """开启overdrive"""
    if game.mecha_overdrive_enabled or game.mecha_overdrive_energy < OVERDRIVE_EN_COST:
        return False
    game.mecha_overdrive_energy -= OVERDRIVE_EN_COST
    game.mecha_overdrive_enabled = True

    # 效果
    if game.mecha_upgrade_total[0] >= 3:
        for i in range(5):
            game.mecha_has_gear[i] = True
    if game.mecha_upgrade_total[0] >= 15:
        for i in range(6):
            if game.persons[i].person_type == PersonType.CARD:
                game.persons[i].is_hint = True
    if game.mecha_upgrade_total[1] >= 15:
        # 拉两个人头（overdrive拉人防死循环上限）
        for _ in range(2):
            for _ in range(OVERDRIVE_INVITE_MAX_TRY):
                if try_invite_people(game, rng):
                    break
    if game.mecha_upgrade_total[2] >= 12:
        game.add_vital(15)
        game.add_motivation(1)

    game.calculate_training_value()
    return True


def mecha_maybe_reverse_overdrive(game) -> bool:
    """恢复overdrive状态（用于重新分配卡组）"""
    if not game.mecha_overdrive_enabled:
        return False
    game.mecha_overdrive_energy += OVERDRIVE_EN_COST
    game.mecha_overdrive_energy = min(game.mecha_overdrive_energy, 6)
    game.mecha_overdrive_enabled = False
    if game.mecha_upgrade_total[2] >= 12:
        game.add_vital(-15)
    return True


def mecha_distribute_en(game, head3: int, chest3: int, foot3: int, rng):
    """分配EN到头胸脚升级"""
    game.mecha_upgrade = [[0] * 3 for _ in range(3)]

    for group_idx, count3 in enumerate([head3, chest3, foot3]):
        en = 3 * count3
        max_item = 5
        while en > 0:
            item = rng.randint(0, 2)
            if game.mecha_upgrade[group_idx][item] < max_item:
                game.mecha_upgrade[group_idx][item] += 1
                en -= 1


def try_invite_people(game, rng) -> bool:
    """尝试拉一个人到训练"""
    invite_person = rng.randint(0, 5)
    invite_train = rng.randint(0, 4)

    space = -1
    for idx in range(MAX_PERSON_PER_TRAIN):
        pid = game.person_distribution[invite_train][idx]
        if pid == -1 and space == -1:
            space = idx
        if pid == invite_person:
            return False

    if space == -1:
        return False

    game.person_distribution[invite_train][space] = invite_person
    return True


def is_card_shining(game, person_idx: int, train_idx: int) -> bool:
    """判断指定卡是否闪彩"""
    if person_idx < 0 or person_idx >= len(game.persons):
        return False
    p = game.persons[person_idx]
    if p.person_type == PersonType.CARD:
        return p.friendship >= 80 and train_idx == p.card_param.card_type
    return False


def init_mecha(game, rng):
    """初始化Dreams剧本（机甲/凯旋门）相关状态"""
    game.mecha_linkeffect_gear_prob_bonus = 0
    game.mecha_linkeffect_lvbonus = False
    game.mecha_rival_lv = [0] * 5
    game.mecha_overdrive_energy = 0
    game.mecha_overdrive_enabled = False
    game.mecha_en = 5
    game.mecha_upgrade = [[0] * 3 for _ in range(3)]
    game.mecha_has_gear = [False] * 5
    game.mecha_win_history = [0] * 5
    game.mecha_any_lose = False

    # Link效果
    for i in range(7):
        chara = game.persons[i].card_param.chara_id if i < 6 else game.uma_id
        if is_link_chara_initial_en(game, chara):
            game.mecha_en += 1
        if is_link_chara_more_gear(game, chara):
            game.mecha_linkeffect_gear_prob_bonus += 1
        if is_link_chara_initial_overdrive(game, chara):
            game.mecha_overdrive_energy += 3
        if is_link_chara_lv_bonus(game, chara):
            game.mecha_linkeffect_lvbonus = True
        if is_link_chara_initial_lv(game, chara):
            for j in range(5):
                game.mecha_rival_lv[j] += 20

    game.mecha_overdrive_energy = min(game.mecha_overdrive_energy, 6)
    game.mecha_en = min(game.mecha_en, 7)
    for i in range(5):
        if game.mecha_rival_lv[i] < 1:
            game.mecha_rival_lv[i] = 1


# ===== Link角色判断 =====

def is_link_chara_initial_en(game, chara_id: int) -> bool:
    """Link角色：初始EN加成（1023=ウオッカ, 1036=スーパークリーク）"""
    if chara_id > 100000:
        chara_id //= 100
    return chara_id in (1023, 1036)


def is_link_chara_more_gear(game, chara_id: int) -> bool:
    """Link角色：齿轮概率加成（1023, 1050, 1084）"""
    if chara_id > 100000:
        chara_id //= 100
    return chara_id in (1023, 1050, 1084)


def is_link_chara_initial_overdrive(game, chara_id: int) -> bool:
    """Link角色：初始overdrive能量（1050, 1083）"""
    if chara_id > 100000:
        chara_id //= 100
    return chara_id in (1050, 1083)


def is_link_chara_lv_bonus(game, chara_id: int) -> bool:
    """Link角色：研究等级倍率加成（1036）"""
    if chara_id > 100000:
        chara_id //= 100
    return chara_id == 1036


def is_link_chara_initial_lv(game, chara_id: int) -> bool:
    """Link角色：初始研究等级（1083, 1084）"""
    if chara_id > 100000:
        chara_id //= 100
    return chara_id in (1083, 1084)
