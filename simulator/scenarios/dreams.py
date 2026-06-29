"""
赛马娘AI训练框架 - Dreams剧本（育马者杯/Beyond Dreams）

scenario_id=13，目前优先实现的剧本。
包含机甲升级、UGE比赛、overdrive等特有机制。

P2-8修复：将game_dreams.py中的函数迁移为DreamsScenario类方法，
Game通过ScenarioBase接口调用。
"""

import random
from typing import List, Optional
from simulator.scenarios.base import ScenarioBase
from simulator.action import GameStage
from simulator.person import PersonType
from config import (
    MAX_PERSON_PER_TRAIN, MECHA_TARGET_TOTAL_LEVEL,
    MECHA_GEAR_PROB, MECHA_GEAR_PROB_LINK_BONUS,
    OVERDRIVE_INVITE_MAX_TRY, OVERDRIVE_EN_COST, OVERDRIVE_TRAIN_MULT,
    URA_START_TURN,
)


class DreamsScenario(ScenarioBase):
    """Dreams剧本（凯旋门/机甲剧本）
    
    特有机制：
    - 研究等级（rivalLv）- 影响训练加成
    - EN点数 - 用于升级机甲（头/胸/脚）
    - UGE比赛 - 定期检查研究等级，给予奖励
    - Overdrive - 消耗能量开启训练强化
    - 齿轮(Gear) - 额外训练加成
    """
    
    SCENARIO_ID = 13
    SCENARIO_NAME = "Beyond Dreams"
    
    # ===== ScenarioBase接口实现 =====

    def on_turn_start(self, game, rng: random.Random):
        """回合开始时的Dreams特定逻辑"""
        pass

    def on_turn_end(self, game, rng: random.Random):
        """回合结束时的Dreams特定逻辑"""
        pass

    def modify_training_value(self, game, train_idx: int, train_value: List[int]) -> List[int]:
        """Dreams剧本训练值修改（乘区已内置在game_calc中）"""
        return train_value

    def get_extra_actions(self, game) -> List[int]:
        """获取Dreams特有的额外动作"""
        return []

    def calculate_score(self, game) -> int:
        """Dreams剧本额外评分（基础评分已由game_score计算，无额外加分）"""
        return 0

    # ===== Dreams专属方法（原game_dreams.py函数） =====

    def maybe_run_uge(self, game) -> bool:
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

    def activate_overdrive(self, game, rng) -> bool:
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
                    if self.try_invite_people(game, rng):
                        break
        if game.mecha_upgrade_total[2] >= 12:
            game.add_vital(15)
            game.add_motivation(1)

        game.calculate_training_value()
        return True

    def maybe_reverse_overdrive(self, game) -> bool:
        """恢复overdrive状态（用于重新分配卡组）"""
        if not game.mecha_overdrive_enabled:
            return False
        game.mecha_overdrive_energy += OVERDRIVE_EN_COST
        game.mecha_overdrive_energy = min(game.mecha_overdrive_energy, 6)
        game.mecha_overdrive_enabled = False
        if game.mecha_upgrade_total[2] >= 12:
            game.add_vital(-15)
        return True

    def distribute_en(self, game, head3: int, chest3: int, foot3: int, rng):
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

    def try_invite_people(self, game, rng) -> bool:
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

    def is_card_shining(self, game, person_idx: int, train_idx: int) -> bool:
        """判断指定卡是否闪彩"""
        if person_idx < 0 or person_idx >= len(game.persons):
            return False
        p = game.persons[person_idx]
        if p.person_type == PersonType.CARD:
            return p.friendship >= 80 and train_idx == p.card_param.card_type
        return False

    def init_mecha(self, game, rng):
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

        # Link效果循环，防止persons长度不足6时越界
        # 支援卡(i<6)和马娘自身(i=6)分开处理
        for i in range(min(6, len(game.persons))):
            chara = game.persons[i].card_param.chara_id
            if self.is_link_chara_initial_en(chara):
                game.mecha_en += 1
            if self.is_link_chara_more_gear(chara):
                game.mecha_linkeffect_gear_prob_bonus += 1
            if self.is_link_chara_initial_overdrive(chara):
                game.mecha_overdrive_energy += 3
            if self.is_link_chara_lv_bonus(chara):
                game.mecha_linkeffect_lvbonus = True
            if self.is_link_chara_initial_lv(chara):
                for j in range(5):
                    game.mecha_rival_lv[j] += 20
        # 马娘自身的Link效果
        chara = game.uma_id
        if self.is_link_chara_initial_en(chara):
            game.mecha_en += 1
        if self.is_link_chara_more_gear(chara):
            game.mecha_linkeffect_gear_prob_bonus += 1
        if self.is_link_chara_initial_overdrive(chara):
            game.mecha_overdrive_energy += 3
        if self.is_link_chara_lv_bonus(chara):
            game.mecha_linkeffect_lvbonus = True
        if self.is_link_chara_initial_lv(chara):
            for j in range(5):
                game.mecha_rival_lv[j] += 20

        game.mecha_overdrive_energy = min(game.mecha_overdrive_energy, 6)
        game.mecha_en = min(game.mecha_en, 7)
        for i in range(5):
            if game.mecha_rival_lv[i] < 1:
                game.mecha_rival_lv[i] = 1

    # ===== Link角色判断 =====

    @staticmethod
    def is_link_chara_initial_en(chara_id: int) -> bool:
        """Link角色：初始EN加成（1023=ウオッカ, 1036=スーパークリーク）"""
        if chara_id > 100000:
            chara_id //= 100
        return chara_id in (1023, 1036)

    @staticmethod
    def is_link_chara_more_gear(chara_id: int) -> bool:
        """Link角色：齿轮概率加成（1023, 1050, 1084）"""
        if chara_id > 100000:
            chara_id //= 100
        return chara_id in (1023, 1050, 1084)

    @staticmethod
    def is_link_chara_initial_overdrive(chara_id: int) -> bool:
        """Link角色：初始overdrive能量（1050, 1083）"""
        if chara_id > 100000:
            chara_id //= 100
        return chara_id in (1050, 1083)

    @staticmethod
    def is_link_chara_lv_bonus(chara_id: int) -> bool:
        """Link角色：研究等级倍率加成（1036）"""
        if chara_id > 100000:
            chara_id //= 100
        return chara_id == 1036

    @staticmethod
    def is_link_chara_initial_lv(chara_id: int) -> bool:
        """Link角色：初始研究等级（1083, 1084）"""
        if chara_id > 100000:
            chara_id //= 100
        return chara_id in (1083, 1084)

    # ===== 辅助静态方法 =====

    @staticmethod
    def get_uge_turns() -> List[int]:
        """获取UGE比赛回合"""
        return [1, 23, 35, 47, 59, 71]

    @staticmethod
    def get_mecha_upgrade_bonus(upgrade_level: int) -> float:
        """机甲升级对应的训练加成"""
        return {0: 0, 1: 10, 2: 18, 3: 26, 4: 33, 5: 40}.get(upgrade_level, 0)
