"""
赛马娘AI训练框架 - Dreams剧本专属模块（thin wrapper）

P2-8修复：逻辑已迁移到DreamsScenario类方法中。
本模块保留为兼容性thin wrapper，新代码应通过game.scenario调用。
"""

from .scenarios.dreams import DreamsScenario

# 单例用于兼容性wrapper调用
_scenario = DreamsScenario()


def mecha_maybe_run_uge(game) -> bool:
    """检查是否触发UGE比赛"""
    return _scenario.maybe_run_uge(game)


def mecha_activate_overdrive(game, rng) -> bool:
    """开启overdrive"""
    return _scenario.activate_overdrive(game, rng)


def mecha_maybe_reverse_overdrive(game) -> bool:
    """恢复overdrive状态"""
    return _scenario.maybe_reverse_overdrive(game)


def mecha_distribute_en(game, head3: int, chest3: int, foot3: int, rng):
    """分配EN到头胸脚升级"""
    return _scenario.distribute_en(game, head3, chest3, foot3, rng)


def try_invite_people(game, rng) -> bool:
    """尝试拉一个人到训练"""
    return _scenario.try_invite_people(game, rng)


def is_card_shining(game, person_idx: int, train_idx: int) -> bool:
    """判断指定卡是否闪彩"""
    return _scenario.is_card_shining(game, person_idx, train_idx)


def init_mecha(game, rng):
    """初始化Dreams剧本相关状态"""
    return _scenario.init_mecha(game, rng)


def is_link_chara_initial_en(game, chara_id: int) -> bool:
    """Link角色：初始EN加成"""
    return DreamsScenario.is_link_chara_initial_en(chara_id)


def is_link_chara_more_gear(game, chara_id: int) -> bool:
    """Link角色：齿轮概率加成"""
    return DreamsScenario.is_link_chara_more_gear(chara_id)


def is_link_chara_initial_overdrive(game, chara_id: int) -> bool:
    """Link角色：初始overdrive能量"""
    return DreamsScenario.is_link_chara_initial_overdrive(chara_id)


def is_link_chara_lv_bonus(game, chara_id: int) -> bool:
    """Link角色：研究等级倍率加成"""
    return DreamsScenario.is_link_chara_lv_bonus(chara_id)


def is_link_chara_initial_lv(game, chara_id: int) -> bool:
    """Link角色：初始研究等级"""
    return DreamsScenario.is_link_chara_initial_lv(chara_id)
