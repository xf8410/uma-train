"""
赛马娘AI训练框架 - Game类

参考 UmaAi 的 Game.h/cpp，Python版游戏模拟器核心。
包含完整游戏状态、训练计算、事件处理、动作应用等。
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from copy import deepcopy

from .action import Action, TrainActionType, GameStage
from .person import (
    Person, PersonType, FriendStage, SupportCard, CardTrainingEffect
)
from simulator.scenarios.base import ScenarioBase
from simulator.scenarios.dreams import DreamsScenario
from config import (
    TOTAL_TURN, MAX_INFO_PERSON_NUM, MAX_PERSON_PER_TRAIN,
    BASIC_FIVE_STATUS_LIMIT, TRAINING_BASIC_VALUE, FAIL_RATE_BASIC,
    RACE_BASIC_FIVE_STATUS_BONUS, RACE_BASIC_PT_BONUS,
    SCORE_PT_RATE_DEFAULT, HINT_LEVEL_PT_RATE_DEFAULT,
    EVENT_PROB, EVENT_STRENGTH_DEFAULT,
    MECHA_LINK_CHARAS, MECHA_GEAR_PROB, MECHA_GEAR_PROB_LINK_BONUS,
    FRIEND_UNLOCK_PROB_LOW_JIBAN, FRIEND_UNLOCK_PROB_HIGH_JIBAN,
    FRIEND_CARD_YAYOI_SSR_ID, FRIEND_CARD_YAYOI_R_ID,
    FRIEND_CARD_LIANGHUA_SSR_ID, FRIEND_CARD_LIANGHUA_R_ID,
    MECHA_TARGET_TOTAL_LEVEL, MECHA_LV_GAIN_BASIC, MECHA_LV_GAIN_SUB_TRAIN_IDX,
    SCORING_NORMAL,
)


# 人员ID常量
PSID_NONE = -1        # 未分配
PSID_NONCARD_YAYOI = 6  # 非卡理事长
PSID_NONCARD_REPORTER = 7  # 非卡记者
PSID_NPC = 8          # NPC

# 友人卡类型
FRIEND_TYPE_NONE = 0
FRIEND_TYPE_LIANGHUA = 1  # 凉花
FRIEND_TYPE_YAYOI = 2     # 理事长(秋川)


@dataclass
class Game:
    """
    赛马娘游戏状态
    
    参考 UmaAi 的 Game.h，用Python dataclass实现。
    包含育成过程中所有状态信息。
    """
    # ===== 参数设置 =====
    pt_score_rate: float = SCORE_PT_RATE_DEFAULT
    hint_pt_rate: float = HINT_LEVEL_PT_RATE_DEFAULT
    event_strength: int = EVENT_STRENGTH_DEFAULT
    scoring_mode: int = SCORING_NORMAL

    # ===== 基本状态 =====
    uma_id: int = 0                    # 马娘编号
    is_link_uma: bool = False          # 是否为link马娘
    is_racing_turn: List[bool] = field(default_factory=lambda: [False] * TOTAL_TURN)
    five_status_bonus: List[int] = field(default_factory=lambda: [0] * 5)  # 成长率
    turn: int = 0                       # 回合数(0-77)
    game_stage: int = GameStage.BEFORE_TRAIN  # 游戏阶段
    vital: int = 100                    # 体力
    max_vital: int = 100                # 体力上限
    motivation: int = 3                 # 干劲(1-5: 绝不调到绝好调)
    five_status: List[int] = field(default_factory=lambda: [0] * 5)   # 五维属性
    five_status_limit: List[int] = field(default_factory=lambda: list(BASIC_FIVE_STATUS_LIMIT))  # 上限
    skill_pt: int = 120                 # 技能点
    skill_score: int = 0                # 已买技能分数
    train_level_count: List[int] = field(default_factory=lambda: [0] * 5)  # 训练等级计数
    failure_rate_bias: int = 0          # 失败率改变量
    is_qie_zhe: bool = False            # 切者
    is_ai_jiao: bool = False            # 爱娇
    is_positive_thinking: bool = False  # 正向思考
    is_refresh_mind: bool = False       # 每回合+5体力

    # 种马
    zhong_ma_blue_count: List[int] = field(default_factory=lambda: [0] * 5)
    zhong_ma_extra_bonus: List[int] = field(default_factory=lambda: [0] * 6)

    is_racing: bool = False             # 当前回合是否在比赛

    # 非卡羁绊
    friendship_noncard_yayoi: int = 0
    friendship_noncard_reporter: int = 0

    # 人头
    persons: List[Person] = field(default_factory=list)
    person_distribution: List[List[int]] = field(default_factory=lambda: [[-1] * MAX_PERSON_PER_TRAIN for _ in range(5)])

    # 赛后加成
    saihou: int = 0

    # 友人卡
    friend_type: int = FRIEND_TYPE_NONE
    friend_is_ssr: bool = False
    friend_person_id: int = PSID_NONE
    friend_stage: int = FriendStage.NOT_CLICKED
    friend_outgoing_used: int = 0
    friend_vital_bonus: float = 1.0
    friend_status_bonus: float = 1.0

    # 得意率
    current_deyilv_bonus: int = 0
    current_lianghua_effect_enable: bool = False

    # ===== Dreams剧本相关 =====
    mecha_linkeffect_gear_prob_bonus: int = 0
    mecha_linkeffect_lvbonus: bool = False
    mecha_rival_lv: List[int] = field(default_factory=lambda: [0] * 5)
    mecha_overdrive_energy: int = 0
    mecha_overdrive_enabled: bool = False
    mecha_en: int = 5
    mecha_upgrade: List[List[int]] = field(default_factory=lambda: [[0] * 3 for _ in range(3)])
    mecha_has_gear: List[bool] = field(default_factory=lambda: [False] * 5)
    mecha_win_history: List[int] = field(default_factory=lambda: [0] * 5)
    mecha_any_lose: bool = False

    # ===== 可计算的非独立信息 =====
    train_value: List[List[int]] = field(default_factory=lambda: [[0] * 6 for _ in range(5)])
    train_vital_change: List[int] = field(default_factory=lambda: [0] * 5)
    fail_rate: List[int] = field(default_factory=lambda: [0] * 5)
    is_train_shining: List[bool] = field(default_factory=lambda: [False] * 5)
    train_value_lower: List[List[int]] = field(default_factory=lambda: [[0] * 6 for _ in range(5)])

    # 中间变量
    mecha_rival_lv_total: int = 0
    mecha_rival_lv_limit: int = 200
    mecha_upgrade_total: List[float] = field(default_factory=lambda: [0.0] * 3)
    mecha_lv_gain: List[List[int]] = field(default_factory=lambda: [[0] * 5 for _ in range(5)])
    mecha_training_status_multiplier: List[float] = field(default_factory=lambda: [1.0] * 6)
    mecha_lv_gain_multiplier: List[float] = field(default_factory=lambda: [1.0] * 5)

    # ===== 方法 =====

    def new_game(self, rng: random.Random, uma_id: int = 0, uma_stars: int = 5,
                 card_ids: Optional[List[int]] = None,
                 zhong_ma_blue: Optional[List[int]] = None,
                 zhong_ma_extra: Optional[List[int]] = None,
                 card_db: Optional[dict] = None):
        """初始化新游戏
        
        Args:
            rng: 随机数生成器
            uma_id: 马娘编号
            uma_stars: 马娘星级
            card_ids: 6张支援卡ID列表
            zhong_ma_blue: 种马蓝因子数量[5]
            zhong_ma_extra: 种马额外加成[6]
            card_db: 支援卡数据库
        """
        if card_ids is None:
            card_ids = [0] * 6
        if zhong_ma_blue is None:
            zhong_ma_blue = [0] * 5
        if zhong_ma_extra is None:
            zhong_ma_extra = [0] * 6

        # 基本状态初始化
        self.turn = 0
        self.game_stage = GameStage.BEFORE_TRAIN
        self.vital = 100
        self.max_vital = 100
        self.motivation = 3
        self.skill_pt = 120
        self.skill_score = 170 * (uma_stars - 2) if uma_stars >= 3 else 120 * uma_stars
        self.failure_rate_bias = 0
        self.is_qie_zhe = False
        self.is_ai_jiao = False
        self.is_positive_thinking = False
        self.is_refresh_mind = False
        self.is_racing = False
        self.friendship_noncard_yayoi = 0
        self.friendship_noncard_reporter = 0
        self.saihou = 0
        self.current_deyilv_bonus = 0
        self.current_lianghua_effect_enable = False

        # 初始化属性
        self.five_status = [0] * 5  # 简化：实际应从马娘数据读取初始值
        self.five_status_limit = list(BASIC_FIVE_STATUS_LIMIT)
        self.train_level_count = [0] * 5

        # 种马
        self.zhong_ma_blue_count = list(zhong_ma_blue)
        self.zhong_ma_extra_bonus = list(zhong_ma_extra)

        # 属性上限和初始值加成（种马）
        for i in range(5):
            self.five_status_limit[i] += int(self.zhong_ma_blue_count[i] * 5.34 * 2)
            self.add_status(i, self.zhong_ma_blue_count[i] * 7)

        # 初始化比赛回合（简化：只有出道赛和URA三连）
        self.is_racing_turn = [False] * TOTAL_TURN
        self.is_racing_turn[11] = True   # 出道赛
        self.is_racing_turn[TOTAL_TURN - 5] = True  # URA1
        self.is_racing_turn[TOTAL_TURN - 3] = True  # URA2
        self.is_racing_turn[TOTAL_TURN - 1] = True  # URA3

        # 初始化支援卡
        self.persons = []
        self.friend_type = FRIEND_TYPE_NONE
        self.friend_is_ssr = False
        self.friend_person_id = PSID_NONE
        self.friend_stage = FriendStage.NOT_CLICKED
        self.friend_outgoing_used = 0
        self.friend_vital_bonus = 1.0
        self.friend_status_bonus = 1.0

        for i in range(MAX_INFO_PERSON_NUM):
            p = Person()
            card_id = card_ids[i] if i < len(card_ids) else 0
            p.set_card(card_id, card_db)
            self.persons.append(p)
            self.saihou += p.card_param.sai_hou

            # 初始加成
            for j in range(5):
                self.add_status(j, p.card_param.initial_bonus[j])
            self.skill_pt += p.card_param.initial_bonus[5]

            # 友人卡识别
            if p.person_type == PersonType.FRIEND_CARD:
                self.friend_person_id = i
                friend_card_id = card_id // 10
                if friend_card_id in (FRIEND_CARD_LIANGHUA_SSR_ID // 10, FRIEND_CARD_LIANGHUA_R_ID // 10):
                    self.friend_type = FRIEND_TYPE_LIANGHUA
                    self.friend_is_ssr = (friend_card_id == FRIEND_CARD_LIANGHUA_SSR_ID // 10)
                elif friend_card_id in (FRIEND_CARD_YAYOI_SSR_ID // 10, FRIEND_CARD_YAYOI_R_ID // 10):
                    self.friend_type = FRIEND_TYPE_YAYOI
                    self.friend_is_ssr = (friend_card_id == FRIEND_CARD_YAYOI_SSR_ID // 10)
                
                self.friend_vital_bonus = 1.0 + 0.01 * p.card_param.event_recovery_amount_up
                self.friend_status_bonus = 1.0 + 0.01 * p.card_param.event_effect_up

        # Dreams剧本初始化
        self._init_mecha(rng)

        # 随机分配卡组并计算训练值
        self.random_distribute_cards(rng)

    def _init_mecha(self, rng: random.Random):
        """初始化Dreams剧本（机甲/凯旋门）相关状态"""
        self.mecha_linkeffect_gear_prob_bonus = 0
        self.mecha_linkeffect_lvbonus = False
        self.mecha_rival_lv = [0] * 5
        self.mecha_overdrive_energy = 0
        self.mecha_overdrive_enabled = False
        self.mecha_en = 5
        self.mecha_upgrade = [[0] * 3 for _ in range(3)]
        self.mecha_has_gear = [False] * 5
        self.mecha_win_history = [0] * 5
        self.mecha_any_lose = False

        # Link效果
        for i in range(7):
            chara = self.persons[i].card_param.chara_id if i < 6 else self.uma_id
            if self._is_link_chara_initial_en(chara):
                self.mecha_en += 1
            if self._is_link_chara_more_gear(chara):
                self.mecha_linkeffect_gear_prob_bonus += 1
            if self._is_link_chara_initial_overdrive(chara):
                self.mecha_overdrive_energy += 3
            if self._is_link_chara_lv_bonus(chara):
                self.mecha_linkeffect_lvbonus = True
            if self._is_link_chara_initial_lv(chara):
                for j in range(5):
                    self.mecha_rival_lv[j] += 20

        self.mecha_overdrive_energy = min(self.mecha_overdrive_energy, 6)
        self.mecha_en = min(self.mecha_en, 7)
        for i in range(5):
            if self.mecha_rival_lv[i] < 1:
                self.mecha_rival_lv[i] = 1

    # ===== 辅助函数 =====

    def is_xiahesu(self) -> bool:
        """是否为夏合宿"""
        return (36 <= self.turn <= 39) or (60 <= self.turn <= 63)

    def is_race_available(self) -> bool:
        """是否可以额外比赛"""
        return 13 <= self.turn <= 71

    def is_end(self) -> bool:
        """是否已经终局"""
        return self.turn >= TOTAL_TURN

    def get_training_level(self, train_idx: int) -> int:
        """计算训练等级（0-4）"""
        if self.is_xiahesu():
            return 4  # 合宿时训练等级固定为4
        return self.train_level_count[train_idx] // 4

    # ===== 属性操作 =====

    def calculate_real_status_gain(self, value: int, gain: int) -> int:
        """考虑1200以上为2的倍数的实际属性增加值"""
        new_value = value + gain
        if new_value <= 1200:
            return gain
        if gain == 1:
            return 2
        return (new_value // 2) * 2 - value

    def add_status(self, idx: int, value: int):
        """增加属性值，处理上限和1200翻倍"""
        t = self.five_status[idx] + value
        t = min(t, self.five_status_limit[idx])
        t = max(t, 1)
        if t > 1200:
            t = (t // 2) * 2
        self.five_status[idx] = t

    def add_all_status(self, value: int):
        """同时增加五个属性值"""
        for i in range(5):
            self.add_status(i, value)

    def add_vital(self, value: int):
        """增加或减少体力"""
        self.vital += value
        self.vital = max(0, min(self.vital, self.max_vital))

    def add_vital_max(self, value: int):
        """增加体力上限"""
        self.max_vital += value
        self.max_vital = min(self.max_vital, 120)

    def add_motivation(self, value: int):
        """增加或减少心情，考虑正向思考"""
        if value < 0:
            if self.is_positive_thinking:
                self.is_positive_thinking = False
            else:
                self.motivation = max(1, self.motivation + value)
        else:
            self.motivation = min(5, self.motivation + value)

    def add_ji_ban(self, idx: int, value: int, ignore_ai_jiao: bool = False):
        """增加羁绊，考虑爱娇"""
        if idx == PSID_NONCARD_YAYOI:
            self.friendship_noncard_yayoi += value
        elif idx == PSID_NONCARD_REPORTER:
            self.friendship_noncard_reporter += value
        elif 0 <= idx < 6:
            gain = (value + 2) if (self.is_ai_jiao and not ignore_ai_jiao) else value
            self.persons[idx].friendship = min(100, self.persons[idx].friendship + gain)
        else:
            # NPC等
            pass

    def add_status_friend(self, idx: int, value: int):
        """友人卡事件，增加属性值或pt"""
        value = int(value * self.friend_status_bonus)
        if idx == 5:
            self.skill_pt += value
        else:
            self.add_status(idx, value)

    def add_vital_friend(self, value: int):
        """友人卡事件，增加体力"""
        value = int(value * self.friend_vital_bonus)
        self.add_vital(value)

    def add_training_level_count(self, train_idx: int, n: int):
        """为某个训练增加n次计数"""
        self.train_level_count[train_idx] = min(16, self.train_level_count[train_idx] + n)

    # ===== 训练计算 =====

    def calculate_failure_rate(self, train_type: int, fail_rate_multiply: float = 1.0) -> int:
        """计算训练失败率
        
        Args:
            train_type: 训练类型(0-4)
            fail_rate_multiply: 失败率乘数
            
        Returns:
            失败率百分比(0-100)
        """
        tlevel = self.get_training_level(train_type)
        x0 = 0.1 * FAIL_RATE_BASIC[train_type][tlevel]
        
        f = 0.0
        if self.vital < x0:
            f = (100 - self.vital) * (x0 - self.vital) / 40.0
        
        f = max(0, min(f, 99))  # 无练习上手最高99%
        f *= fail_rate_multiply
        
        fr = math.ceil(f) + self.failure_rate_bias
        return max(0, min(fr, 100))

    def calculate_training_value_single(self, tra: int):
        """计算单个训练的数值
        
        参考 Game.cpp 的 calculateTrainingValueSingle
        """
        head_num = 0      # 人头数
        shining_num = 0   # 闪彩数
        link_num = 0      # link卡数

        tlevel = self.get_training_level(tra)

        is_card_shining_record = [False] * 6

        # 统计人头和闪彩
        for h in range(MAX_PERSON_PER_TRAIN):
            p_idx = self.person_distribution[tra][h]
            if p_idx < 0:
                break
            if p_idx == PSID_NPC:
                head_num += 1
                continue
            if p_idx >= 6:
                continue  # 非支援卡

            head_num += 1
            p = self.persons[p_idx]
            if self.is_card_shining(p_idx, tra):
                shining_num += 1
                is_card_shining_record[p_idx] = True
            if p.card_param.is_link:
                link_num += 1

        self.is_train_shining[tra] = shining_num > 0

        # 基础值
        basic_value = list(TRAINING_BASIC_VALUE[tra][tlevel])
        vital_cost_basic = -basic_value[6]  # 体力消耗（取正值）

        # 累计支援卡效果
        total_xun_lian = 0
        total_gan_jing = 0
        total_you_qing_multiplier = 1.0
        vital_cost_multiplier = 1.0
        fail_rate_multiplier = 1.0

        for h in range(MAX_PERSON_PER_TRAIN):
            pid = self.person_distribution[tra][h]
            if pid < 0:
                break
            if pid >= 6:
                continue

            p = self.persons[pid]
            is_this_shining = is_card_shining_record[pid]
            
            eff = p.card_param.get_card_effect(
                is_this_shining, tra, p.friendship, p.card_record,
                head_num, shining_num
            )

            # 基础值bonus
            for i in range(6):
                if basic_value[i] > 0:
                    basic_value[i] += int(eff.bonus[i])

            # 闪彩效果
            if is_card_shining_record[pid]:
                total_you_qing_multiplier *= (1 + 0.01 * eff.you_qing)
                if tra == TrainActionType.WIT:
                    vital_cost_basic -= eff.vital_bonus

            total_xun_lian += eff.xun_lian
            total_gan_jing += eff.gan_jing
            vital_cost_multiplier *= (1 - 0.01 * eff.vital_cost_drop)
            fail_rate_multiplier *= (1 - 0.01 * eff.fail_rate_drop)

        # overdrive体力消耗减半
        if self.mecha_overdrive_enabled and self.mecha_upgrade_total[0] >= 15:
            vital_cost_multiplier *= 0.5

        # 体力变化
        vital_change_int = -int(vital_cost_basic * vital_cost_multiplier) if vital_cost_basic > 0 else -vital_cost_basic
        vital_change_int = max(-self.vital, min(vital_change_int, self.max_vital - self.vital))
        self.train_vital_change[tra] = vital_change_int
        self.fail_rate[tra] = self.calculate_failure_rate(tra, fail_rate_multiplier)

        # 支援卡乘区
        card_multiplier = (
            (1 + 0.05 * head_num) *
            (1 + 0.01 * total_xun_lian) *
            (1 + 0.1 * (self.motivation - 3) * (1 + 0.01 * total_gan_jing)) *
            total_you_qing_multiplier
        )

        # 下层值
        for i in range(6):
            is_related = basic_value[i] != 0
            bvl = basic_value[i]
            uma_bonus = 1 + 0.01 * self.five_status_bonus[i] if i < 5 else 1
            self.train_value_lower[tra][i] = bvl * card_multiplier * uma_bonus

        # 彩圈必有齿轮
        if shining_num > 0:
            self.mecha_has_gear[tra] = True

        # 上层（剧本乘区）
        scenario_train_multiplier = 1.0

        # 研究等级加成
        lv_bonus = (6 + 0.06 * self.mecha_rival_lv[tra]) if self.mecha_rival_lv[tra] > 1 else 0
        if self.mecha_linkeffect_lvbonus:
            lv_bonus *= 1.5
        scenario_train_multiplier *= (1 + 0.01 * lv_bonus)

        # 齿轮加成
        if self.mecha_has_gear[tra]:
            if self.turn < 12:
                gear_bonus = 3
            elif self.turn < 24:
                gear_bonus = 6
            elif self.turn < 36:
                gear_bonus = 10
            elif self.turn < 48:
                gear_bonus = 16
            elif self.turn < 60:
                gear_bonus = 20
            elif self.turn < 72:
                gear_bonus = 25
            else:
                gear_bonus = 30
            scenario_train_multiplier *= (1 + 0.01 * gear_bonus)

        # 胸3号友情加成
        if shining_num > 0:
            friendship_bonus = 2 * self.mecha_upgrade[1][2]
            scenario_train_multiplier *= (1 + 0.01 * friendship_bonus)

        # overdrive加成
        if self.mecha_overdrive_enabled:
            scenario_train_multiplier *= 1.25
            head_bonus = 1 if self.mecha_upgrade_total[1] >= 3 else (3 if self.mecha_upgrade_total[1] >= 12 else 0)
            scenario_train_multiplier *= (1 + 0.01 * head_num * head_bonus)

        # 计算最终值
        for i in range(6):
            lower = self.train_value_lower[tra][i]
            lower = min(lower, 100)
            self.train_value_lower[tra][i] = lower
            
            total = int(lower * scenario_train_multiplier * self.mecha_training_status_multiplier[i])
            upper = total - lower
            upper = min(upper, 100)
            
            if i < 5:
                lower = self.calculate_real_status_gain(self.five_status[i], lower)
                upper = self.calculate_real_status_gain(self.five_status[i] + lower, upper)
            
            self.train_value[tra][i] = upper + lower

        # 研究等级提升量
        self.calculate_lv_gain_single(tra, head_num, shining_num > 0)

    def calculate_lv_gain_single(self, tra: int, head_num: int, is_shining: bool):
        """计算每个训练加多少研究等级"""
        xhs = self.is_xiahesu()
        group = 0 if not self.mecha_has_gear[tra] else (1 if not is_shining else 2)
        
        for i in range(5):
            self.mecha_lv_gain[tra][i] = 0
        
        for sub in range(3):
            train_type = MECHA_LV_GAIN_SUB_TRAIN_IDX[tra][sub]
            basic = MECHA_LV_GAIN_BASIC[int(xhs)][group][sub][head_num]
            multiplier = self.mecha_lv_gain_multiplier[train_type]
            gain = int(multiplier * basic)
            if gain == basic and multiplier > 1:
                gain += 1
            self.mecha_lv_gain[tra][train_type] = gain

    def calculate_training_value(self):
        """计算所有训练分别加多少"""
        # 重新计算统计信息
        self.mecha_rival_lv_total = sum(self.mecha_rival_lv)
        
        # 研究等级上限
        if self.turn < 24:
            self.mecha_rival_lv_limit = 200
        elif self.turn < 36:
            self.mecha_rival_lv_limit = 300
        elif self.turn < 48:
            self.mecha_rival_lv_limit = 400
        elif self.turn < 60:
            self.mecha_rival_lv_limit = 500
        elif self.turn < 72:
            self.mecha_rival_lv_limit = 600
        else:
            self.mecha_rival_lv_limit = 700

        for i in range(3):
            self.mecha_upgrade_total[i] = sum(self.mecha_upgrade[i])

        # 属性加成倍率
        for i in range(5):
            m = 1.0
            if self.mecha_overdrive_enabled:
                upgrade_group = (
                    self.mecha_upgrade_total[2] if i in (0, 2) else
                    self.mecha_upgrade_total[1] if i in (1, 3) else
                    self.mecha_upgrade_total[0]
                )
                if upgrade_group >= 9:
                    count = 1 + (self.mecha_rival_lv_total - 1) // 200
                    m *= (1 + 0.03 * count)
                elif upgrade_group >= 6:
                    count = 1 + (self.mecha_rival_lv_total - 1) // 300
                    m *= (1 + 0.03 * count)
            self.mecha_training_status_multiplier[i] = m

        # pt倍率
        ptb = 1.0 * (1 + self.mecha_upgrade[2][2] * 0.12)
        if self.mecha_overdrive_enabled and self.mecha_upgrade_total[2] >= 15:
            count = 1 + (self.mecha_rival_lv_total - 1) // 150
            ptb *= (1 + 0.03 * count)
        self.mecha_training_status_multiplier[5] = ptb

        # 研究等级提升量倍率
        for i in range(5):
            upgrade_lv = (
                self.mecha_upgrade[2][0] if i == 0 else
                self.mecha_upgrade[1][0] if i == 1 else
                self.mecha_upgrade[2][1] if i == 2 else
                self.mecha_upgrade[1][1] if i == 3 else
                self.mecha_upgrade[0][0]
            )
            lv_gain_bonus = {5: 40, 4: 33, 3: 26, 2: 18, 1: 10}.get(upgrade_lv, 0)
            if self.mecha_overdrive_enabled:
                if self.mecha_upgrade_total[0] >= 12:
                    lv_gain_bonus += 25
                elif self.mecha_upgrade_total[0] >= 9:
                    lv_gain_bonus += 20
                elif self.mecha_upgrade_total[0] >= 6:
                    lv_gain_bonus += 15
            self.mecha_lv_gain_multiplier[i] = 1.0 + 0.01 * lv_gain_bonus

        for i in range(5):
            self.calculate_training_value_single(i)

    # ===== 卡组分配 =====

    def random_distribute_cards(self, rng: random.Random):
        """随机分配人头到训练"""
        self.person_distribution = [[-1] * MAX_PERSON_PER_TRAIN for _ in range(5)]

        if self.is_racing:
            return

        # overdrive恢复
        overdrive_was_enabled = self._mecha_maybe_reverse_overdrive()

        head_n = [0] * 5
        buckets = [[] for _ in range(5)]

        # 先放友人/理事长/记者
        for i in range(8):
            at_train = 5  # 默认不出现
            if self.friend_type != FRIEND_TYPE_NONE and i == self.friend_person_id:
                at_train = self.persons[i].get_deyilv_train(rng)
            elif i == PSID_NONCARD_YAYOI and self.friend_type != FRIEND_TYPE_YAYOI:
                at_train = self._noncard_distribution(rng)
            elif i == PSID_NONCARD_REPORTER:
                if self.turn < 12 or self.is_xiahesu():
                    continue
                at_train = self._noncard_distribution(rng)

            if at_train < 5:
                buckets[at_train].append(i)

        for i in range(5):
            if len(buckets[i]) == 1:
                self.person_distribution[i][0] = buckets[i][0]
                head_n[i] += 1
            elif len(buckets[i]) > 1:
                self.person_distribution[i][0] = buckets[i][rng.randint(0, len(buckets[i]) - 1)]
                head_n[i] += 1
            buckets[i] = []

        # 普通支援卡
        for i in range(6):
            p = self.persons[i]
            if p.person_type == PersonType.CARD:
                at_train = p.get_deyilv_train(rng)
                if at_train < 5:
                    buckets[at_train].append(i)

        # NPC
        for _ in range(6):
            at_train = self._npc_distribution(rng)
            if at_train < 5:
                buckets[at_train].append(PSID_NPC)

        # 选出不超过5个人头
        for i in range(5):
            max_head = 5 - head_n[i]
            if len(buckets[i]) <= max_head:
                for j in range(len(buckets[i])):
                    self.person_distribution[i][head_n[i]] = buckets[i][j]
                    head_n[i] += 1
            else:
                # 随机选max_head个
                selected = rng.sample(buckets[i], max_head)
                for j, pid in enumerate(selected):
                    self.person_distribution[i][head_n[i]] = pid
                    head_n[i] += 1

        # 是否有hint
        for pid in range(6):
            if self.persons[pid].person_type == PersonType.CARD:
                hint_prob = 0.06 * (1 + 0.01 * self.persons[pid].card_param.hint_prob_increase)
                hint_prob *= (1.0 + 0.15 * self.mecha_upgrade[0][1])
                self.persons[pid].is_hint = rng.random() < hint_prob

        # 随机决定是否有齿轮
        gear_prob = MECHA_GEAR_PROB + MECHA_GEAR_PROB_LINK_BONUS * self.mecha_linkeffect_gear_prob_bonus
        for i in range(5):
            self.mecha_has_gear[i] = rng.random() < gear_prob

        # URA期间自动开启overdrive
        if self.turn >= 72 and not self.mecha_any_lose:
            self.mecha_overdrive_energy = 3
            self._mecha_activate_overdrive(rng)
        elif overdrive_was_enabled:
            self._mecha_activate_overdrive(rng)

        self.calculate_training_value()

    def _noncard_distribution(self, rng: random.Random) -> int:
        """非卡理事长/记者的分布"""
        probs = [100, 100, 100, 100, 100, 200]
        return self._weighted_choice(rng, probs)

    def _npc_distribution(self, rng: random.Random) -> int:
        """NPC的分布"""
        probs = [100, 100, 100, 100, 100, 100]
        return self._weighted_choice(rng, probs)

    @staticmethod
    def _weighted_choice(rng: random.Random, weights: list) -> int:
        """加权随机选择"""
        total = sum(weights)
        r = rng.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r < cumulative:
                return i if i < 5 else 5
        return min(len(weights) - 1, 5)

    # ===== 动作相关 =====

    def is_legal(self, action: Action) -> bool:
        """判断动作是否合法"""
        if not action.is_standard() and action.to_int() < 0:
            return False

        if action.type != self.game_stage:
            return False

        if action.type == GameStage.BEFORE_TRAIN:
            if self.is_racing:
                return action.train == TrainActionType.RACE

            if action.overdrive:
                if self.mecha_overdrive_energy < 3:
                    return False
                if self.mecha_overdrive_enabled:
                    return False
                if self.mecha_upgrade_total[1] >= 15:
                    return action.train == -1
                else:
                    return 0 <= action.train <= 4

            if action.train == TrainActionType.REST:
                return not self.is_xiahesu()
            elif action.train == TrainActionType.OUTGOING:
                return True
            elif action.train == TrainActionType.RACE:
                return self.is_race_available()
            elif 0 <= action.train <= 4:
                return True
            return False

        elif action.type == GameStage.BEFORE_MECHA_UPGRADE:
            total3 = self.mecha_en // 3
            mecha_head_limit = 5 if self.turn >= 35 else 3
            mecha_chest_limit = 5 if self.turn >= 59 else 3
            mecha_foot = total3 - action.mecha_head - action.mecha_chest
            mecha_foot_limit = 5 if self.turn >= 59 else 3
            if action.mecha_head < 0 or action.mecha_head > mecha_head_limit:
                return False
            if action.mecha_chest < 0 or action.mecha_chest > mecha_chest_limit:
                return False
            if mecha_foot < 0 or mecha_foot > mecha_foot_limit:
                return False
            return True

        return False

    def apply_action(self, rng: random.Random, action: Action):
        """应用动作，推进游戏状态"""
        if self.is_end():
            return

        if action.type == GameStage.BEFORE_MECHA_UPGRADE:
            mecha_foot = self.mecha_en // 3 - action.mecha_head - action.mecha_chest
            self._mecha_distribute_en(action.mecha_head, action.mecha_chest, mecha_foot, rng)
            self.game_stage = GameStage.BEFORE_TRAIN
            self._check_event_after_train(rng)
            if not self.is_end():
                self.random_distribute_cards(rng)
        else:
            if action.overdrive:
                self._mecha_activate_overdrive(rng)
            if action.train != TrainActionType.NONE:
                self._apply_training(rng, action.train)
                run_uge = self._mecha_maybe_run_uge()
                if not run_uge:
                    self._check_event_after_train(rng)
                    if not self.is_end():
                        self.random_distribute_cards(rng)

    def _apply_training(self, rng: random.Random, train: int) -> bool:
        """处理训练/出行/比赛"""
        if train == TrainActionType.REST:
            # 休息：回复50体力（夏合宿时通过外出选项实现）
            self.add_vital(50)
            if rng.random() < 0.04:
                self.add_motivation(1)
            return True
        elif train == TrainActionType.OUTGOING:
            # 外出
            if self.friend_type != FRIEND_TYPE_NONE and self.friend_stage >= FriendStage.AFTER_UNLOCK_OUTGOING and self.friend_outgoing_used < 5:
                self._handle_friend_outgoing(rng)
            else:
                # 普通外出
                self.add_vital(10)
                self.add_motivation(1)
            return True
        elif train == TrainActionType.RACE:
            # 比赛
            self._run_race(RACE_BASIC_FIVE_STATUS_BONUS, RACE_BASIC_PT_BONUS)
            return True
        elif 0 <= train <= 4:
            # 训练
            fail_rate = self.fail_rate[train]
            roll = rng.randint(1, 100)
            
            if roll <= fail_rate:
                # 训练失败
                if fail_rate >= 20:
                    # 大失败
                    for i in range(5):
                        self.add_status(i, -4)
                        if self.five_status[i] > 1200:
                            self.add_status(i, -4)
                    self.add_motivation(-3)
                    self.add_vital(10)
                else:
                    # 小失败
                    self.add_status(train, -5)
                    if self.five_status[train] > 1200:
                        self.add_status(train, -5)
                    self.add_motivation(-1)
            else:
                # 成功
                for i in range(5):
                    self.add_status(i, self.train_value[train][i])
                self.skill_pt += self.train_value[train][5]
                self.add_vital(self.train_vital_change[train])

                # 羁绊增加
                friendship_extra = 0
                if self.mecha_overdrive_enabled and self.mecha_upgrade_total[2] >= 3:
                    friendship_extra += 3
                is_ssr_yayoi = self.friend_type == FRIEND_TYPE_YAYOI and self.friend_is_ssr
                if is_ssr_yayoi:
                    friendship_extra += 1

                hint_cards = []
                click_friend = False

                # 检查SSR友人
                for h in range(MAX_PERSON_PER_TRAIN):
                    p = self.person_distribution[train][h]
                    if p < 0:
                        break
                    if is_ssr_yayoi and p == self.friend_person_id:
                        friendship_extra += 2
                        break

                for h in range(MAX_PERSON_PER_TRAIN):
                    p = self.person_distribution[train][h]
                    if p < 0:
                        break

                    if p == self.friend_person_id and self.friend_type != FRIEND_TYPE_NONE:
                        self.add_ji_ban(p, 4 + friendship_extra)
                        click_friend = True
                    elif p < 6:
                        self.add_ji_ban(p, 7 + friendship_extra)
                        if self.persons[p].is_hint:
                            hint_cards.append(p)
                    elif p == PSID_NPC:
                        pass
                    elif p == PSID_NONCARD_YAYOI:
                        jiban = self.friendship_noncard_yayoi
                        g = 2 if jiban < 40 else (3 if jiban < 60 else (4 if jiban < 80 else 5))
                        self.skill_pt += g
                        self.add_ji_ban(PSID_NONCARD_YAYOI, 7)
                    elif p == PSID_NONCARD_REPORTER:
                        jiban = self.friendship_noncard_reporter
                        g = 2 if jiban < 40 else (3 if jiban < 60 else (4 if jiban < 80 else 5))
                        self.add_status(train, g)
                        self.add_ji_ban(PSID_NONCARD_REPORTER, 7)

                # hint处理
                if hint_cards:
                    if not (self.mecha_overdrive_enabled and self.mecha_upgrade_total[0] >= 15):
                        hint_card = rng.choice(hint_cards)
                        hint_cards = [hint_card]
                    
                    for hc in hint_cards:
                        self.add_ji_ban(hc, 5)
                        hint_level = self.persons[hc].card_param.hint_level
                        if hint_level > 0:
                            self.skill_pt += int(hint_level * self.hint_pt_rate)
                        else:
                            # 根乌拉拉等只给属性
                            hint_bonus = [
                                [6, 0, 2, 0, 0],  # 速训练
                                [0, 6, 0, 2, 0],  # 耐训练
                                [0, 2, 6, 0, 0],  # 力训练
                                [1, 0, 1, 6, 0],  # 根训练
                                [0, 0, 0, 0, 6],  # 智训练
                            ]
                            for i, v in enumerate(hint_bonus[train]):
                                self.add_status(i, v)
                            if train == 4:
                                self.skill_pt += 5

                # 友人点击事件
                if click_friend:
                    self._handle_friend_click_event(rng, train)

                # 训练等级提升
                self.add_training_level_count(train, 1)

                # 齿轮能量
                if self.mecha_has_gear[train]:
                    self.mecha_overdrive_energy = min(6, self.mecha_overdrive_energy + 1)

                # 研究等级提升
                for i in range(5):
                    self.mecha_rival_lv[i] = min(
                        self.mecha_rival_lv_limit,
                        self.mecha_rival_lv[i] + self.mecha_lv_gain[train][i]
                    )

            return True
        return False

    def _run_race(self, basic_status_bonus: int, basic_pt_bonus: int):
        """比赛奖励"""
        race_multiply = 1 + 0.01 * self.saihou
        status_bonus = int(race_multiply * basic_status_bonus)
        pt_bonus = int(race_multiply * basic_pt_bonus)
        self.add_all_status(status_bonus)
        self.skill_pt += pt_bonus

    # ===== 友人卡相关 =====

    def _handle_friend_click_event(self, rng: random.Random, at_train: int):
        """友人点击事件"""
        if self.friend_type == FRIEND_TYPE_NONE:
            return

        if self.friend_stage == FriendStage.NOT_CLICKED:
            self.friend_stage = FriendStage.BEFORE_UNLOCK_OUTGOING
            if self.friend_type == FRIEND_TYPE_YAYOI:
                self.add_status_friend(0, 8)
                self.add_ji_ban(self.friend_person_id, 10)
                self.add_motivation(1)
            elif self.friend_type == FRIEND_TYPE_LIANGHUA:
                self.add_vital_max(4)
                self.add_ji_ban(self.friend_person_id, 10)
                self.add_motivation(1)

    def _handle_friend_outgoing(self, rng: random.Random):
        """友人外出"""
        if self.friend_type == FRIEND_TYPE_YAYOI:
            outings = [
                # (体力回复, 干劲, 属性加成, pt加成)
                lambda: (self.add_vital_friend(30), self.add_motivation(1), self.add_status_friend(3, 20), self.add_ji_ban(self.friend_person_id, 5)),
                lambda: (self.add_vital_friend(30), self.add_motivation(1), self.add_status_friend(0, 10), self.add_status_friend(3, 10), setattr(self, 'is_refresh_mind', True), self.add_ji_ban(self.friend_person_id, 5)),
                lambda: (self.add_vital_friend(43) if self.max_vital - self.vital >= 20 else self.add_status_friend(3, 29), self.add_motivation(1), self.add_ji_ban(self.friend_person_id, 5)),
                lambda: (self.add_vital_friend(30), self.add_motivation(1), self.add_status_friend(3, 25), self.add_ji_ban(self.friend_person_id, 5)),
                lambda: (self.add_vital_friend(30) if rng.random() < 0.75 else self.add_vital_friend(26),
                         self.add_status_friend(3, 36 if rng.random() < 0.75 else 24),
                         setattr(self, 'skill_pt', self.skill_pt + (72 if rng.random() < 0.75 else 40)),
                         self.add_motivation(1), self.add_ji_ban(self.friend_person_id, 5),
                         setattr(self, 'is_refresh_mind', True) if rng.random() < 0.75 else None),
            ]
            if self.friend_outgoing_used < len(outings):
                outings[self.friend_outgoing_used]()

        elif self.friend_type == FRIEND_TYPE_LIANGHUA:
            if self.friend_outgoing_used == 0:
                self.add_vital_friend(35)
                self.add_motivation(1)
                self.add_status_friend(0, 15)
                self.add_ji_ban(self.friend_person_id, 5)
            elif self.friend_outgoing_used == 1:
                self.add_vital_friend(30)
                self.add_motivation(1)
                self.add_status_friend(0, 10)
                self.add_status_friend(4, 10)
                self.add_ji_ban(self.friend_person_id, 5)
            elif self.friend_outgoing_used == 2:
                self.add_vital_friend(50)
                self.add_motivation(1)
                self.add_ji_ban(self.friend_person_id, 5)
            elif self.friend_outgoing_used == 3:
                self.add_vital_friend(30)
                self.add_motivation(1)
                self.add_status_friend(0, 25)
                self.add_ji_ban(self.friend_person_id, 5)
            elif self.friend_outgoing_used == 4:
                if rng.random() < 0.75:
                    self.add_vital_friend(40)
                    self.add_status_friend(0, 30)
                    self.skill_pt += 72
                else:
                    self.add_vital_friend(35)
                    self.add_status_friend(0, 15)
                    self.skill_pt += 40

        self.friend_outgoing_used += 1

    def _handle_friend_unlock(self, rng: random.Random):
        """友人外出解锁"""
        if self.friend_type == FRIEND_TYPE_YAYOI:
            if self.max_vital - self.vital >= 15:
                self.add_vital_friend(25)
            else:
                self.add_status_friend(0, 8)
                self.add_status_friend(3, 8)
                self.skill_pt += 10
            self.add_motivation(1)
            self.is_refresh_mind = True
            self.add_ji_ban(self.friend_person_id, 5)
        elif self.friend_type == FRIEND_TYPE_LIANGHUA:
            self.add_vital_max(4)
            self.add_vital_friend(20)
            self.add_motivation(1)
            self.add_ji_ban(self.friend_person_id, 5)

        self.friend_stage = FriendStage.AFTER_UNLOCK_OUTGOING

    # ===== 事件处理 =====

    def _check_event_after_train(self, rng: random.Random):
        """训练后检查事件并推进回合"""
        self.mecha_overdrive_enabled = False
        self._maybe_update_deyilv()
        self._check_fixed_events(rng)
        self._check_random_events(rng)

        # 回合+1
        self.turn += 1
        if self.turn < TOTAL_TURN:
            self.is_racing = self.is_racing_turn[self.turn]
        self.game_stage = GameStage.BEFORE_TRAIN

    def _maybe_update_deyilv(self):
        """更新得意率"""
        deyilv_bonus = 15 * self.mecha_upgrade[0][0]
        lianghua_enable = (
            self.friend_type == FRIEND_TYPE_LIANGHUA and
            self.friend_is_ssr and
            self.persons[self.friend_person_id].friendship >= 60 if self.friend_person_id >= 0 else False
        )
        if deyilv_bonus != self.current_deyilv_bonus or lianghua_enable != self.current_lianghua_effect_enable:
            self.current_deyilv_bonus = deyilv_bonus
            self.current_lianghua_effect_enable = lianghua_enable
            for i in range(6):
                self.persons[i].set_extra_deyilv_bonus(deyilv_bonus, lianghua_enable)

    def _check_fixed_events(self, rng: random.Random):
        """每回合的固定事件"""
        if self.is_refresh_mind:
            self.add_vital(5)
            if rng.random() < 0.25:
                self.is_refresh_mind = False

        if self.is_racing:
            if self.turn < 72:
                self._run_race(3, 45)
                self.add_ji_ban(PSID_NONCARD_YAYOI if self.friend_type != FRIEND_TYPE_YAYOI else self.friend_person_id, 4, ignore_ai_jiao=True)
            elif self.turn == 73:
                self._run_race(10, 40)
            elif self.turn == 75:
                self._run_race(10, 60)
            elif self.turn == 77:
                self._run_race(10, 80)

        if self.turn == 23:  # 第一年年底
            vital_space = self.max_vital - self.vital
            if vital_space >= 20:
                self.add_vital(20)
            else:
                self.add_all_status(5)

        elif self.turn == 29:  # 第二年继承
            for i in range(5):
                self.add_status(i, self.zhong_ma_blue_count[i] * 6)
            factor = rng.random() * 2
            for i in range(5):
                self.add_status(i, int(factor * self.zhong_ma_extra_bonus[i]))
            self.skill_pt += int((0.5 + 0.5 * factor) * self.zhong_ma_extra_bonus[5])
            for i in range(5):
                self.five_status_limit[i] += self.zhong_ma_blue_count[i] * 2
                self.five_status_limit[i] += rng.randint(0, 7)

        elif self.turn == 47:  # 第二年年底
            vital_space = self.max_vital - self.vital
            if vital_space >= 30:
                self.add_vital(30)
            else:
                self.add_all_status(8)

        elif self.turn == 48:  # 抽奖
            rd = rng.randint(0, 99)
            if rd < 16:
                self.add_vital(30)
                self.add_all_status(10)
                self.add_motivation(2)
            elif rd < 43:
                self.add_vital(20)
                self.add_all_status(5)
                self.add_motivation(1)
            elif rd < 89:
                self.add_vital(20)
            else:
                self.add_motivation(-1)

        elif self.turn == 49:
            self.skill_score += 170

        elif self.turn == 53:  # 第三年继承
            for i in range(5):
                self.add_status(i, self.zhong_ma_blue_count[i] * 6)
            factor = rng.random() * 2
            for i in range(5):
                self.add_status(i, int(factor * self.zhong_ma_extra_bonus[i]))
            self.skill_pt += int((0.5 + 0.5 * factor) * self.zhong_ma_extra_bonus[5])
            for i in range(5):
                self.five_status_limit[i] += self.zhong_ma_blue_count[i] * 2
                self.five_status_limit[i] += rng.randint(0, 7)

        elif self.turn == 70:
            self.skill_score += 170

        elif self.turn == 77:  # URA3结算
            # 记者
            if self.friendship_noncard_reporter >= 80:
                self.add_all_status(5)
                self.skill_pt += 20
            elif self.friendship_noncard_reporter >= 60:
                self.add_all_status(3)
                self.skill_pt += 10
            elif self.friendship_noncard_reporter >= 40:
                self.skill_pt += 10
            else:
                self.skill_pt += 5

            # 全胜检查
            if all(h == 2 for h in self.mecha_win_history):
                self.skill_pt += 40
                self.add_all_status(45)
                self.skill_pt += 175
            else:
                self.add_all_status(40)

            self.add_all_status(5)
            self.skill_pt += 20

    def _check_random_events(self, rng: random.Random):
        """模拟随机事件"""
        if self.turn >= 72:
            return

        # 友人解锁出行
        if self.friend_type != FRIEND_TYPE_NONE:
            p = self.persons[self.friend_person_id]
            if self.friend_stage == FriendStage.BEFORE_UNLOCK_OUTGOING:
                unlock_prob = (
                    FRIEND_UNLOCK_PROB_HIGH_JIBAN if p.friendship >= 60
                    else FRIEND_UNLOCK_PROB_LOW_JIBAN
                )
                if rng.random() < unlock_prob:
                    self._handle_friend_unlock(rng)

        # 支援卡连续事件
        if rng.random() < EVENT_PROB:
            card = rng.randint(0, 5)
            self.add_ji_ban(card, 5)
            self.add_status(rng.randint(0, 4), self.event_strength)
            self.skill_pt += self.event_strength
            if rng.random() < 0.4 * (1.0 - self.turn / TOTAL_TURN):
                self.add_motivation(1)
            if rng.random() < 0.5:
                self.add_vital(10)
            elif rng.random() < 0.06:
                self.add_vital(-10)

        # 马娘随机事件
        if rng.random() < 0.1:
            self.add_all_status(3)

        # 体力事件
        if rng.random() < 0.10:
            self.add_vital(5)
        if rng.random() < 0.02:
            self.add_vital(30)

        # 心情事件
        if rng.random() < 0.02:
            self.add_motivation(1)
        if self.turn >= 12 and rng.random() < 0.04:
            self.add_motivation(-1)

    # ===== Dreams剧本相关 =====

    def _mecha_maybe_run_uge(self) -> bool:
        """检查是否触发UGE比赛"""
        uge_turns = [1, 23, 35, 47, 59, 71]
        if self.turn not in uge_turns:
            return False

        self.game_stage = GameStage.BEFORE_MECHA_UPGRADE
        if self.turn == 1:
            return True

        uge_count = self.turn // 12 - 1
        total_lv = sum(self.mecha_rival_lv)
        target = MECHA_TARGET_TOTAL_LEVEL[uge_count]

        if total_lv >= target:
            # S评价
            self.mecha_en += 6
            self.add_all_status(10 + 5 * uge_count)
            self.skill_pt += 25 + 10 * uge_count
            if uge_count in (0, 2, 4):
                for i in range(5):
                    self.add_training_level_count(i, 4)
            self.mecha_win_history[uge_count] = 2
        elif total_lv >= target * 7 // 10:
            # A评价
            self.mecha_en += 5
            self.add_all_status(10 + 5 * uge_count)
            self.skill_pt += 25 + 10 * uge_count
            if uge_count in (0, 2, 4):
                for i in range(5):
                    self.add_training_level_count(i, 4)
            self.mecha_win_history[uge_count] = 1
            self.mecha_any_lose = True
        else:
            # B评价
            self.mecha_en += 4
            self.add_all_status(5 + 5 * uge_count)
            self.skill_pt += 20 + 10 * uge_count
            self.mecha_win_history[uge_count] = 0
            self.mecha_any_lose = True

        return True

    def _mecha_activate_overdrive(self, rng: random.Random) -> bool:
        """开启overdrive"""
        if self.mecha_overdrive_enabled or self.mecha_overdrive_energy < 3:
            return False
        self.mecha_overdrive_energy -= 3
        self.mecha_overdrive_enabled = True

        # 效果
        if self.mecha_upgrade_total[0] >= 3:
            for i in range(5):
                self.mecha_has_gear[i] = True
        if self.mecha_upgrade_total[0] >= 15:
            for i in range(6):
                if self.persons[i].person_type == PersonType.CARD:
                    self.persons[i].is_hint = True
        if self.mecha_upgrade_total[1] >= 15:
            # 拉两个人头
            for _ in range(2):
                for _ in range(1000):
                    if self._try_invite_people(rng):
                        break
        if self.mecha_upgrade_total[2] >= 12:
            self.add_vital(15)
            self.add_motivation(1)

        self.calculate_training_value()
        return True

    def _mecha_maybe_reverse_overdrive(self) -> bool:
        """恢复overdrive状态（用于重新分配卡组）"""
        if not self.mecha_overdrive_enabled:
            return False
        self.mecha_overdrive_energy += 3
        self.mecha_overdrive_energy = min(self.mecha_overdrive_energy, 6)
        self.mecha_overdrive_enabled = False
        if self.mecha_upgrade_total[2] >= 12:
            self.add_vital(-15)
        return True

    def _mecha_distribute_en(self, head3: int, chest3: int, foot3: int, rng: random.Random):
        """分配EN到头胸脚升级"""
        self.mecha_upgrade = [[0] * 3 for _ in range(3)]

        for group_idx, count3 in enumerate([head3, chest3, foot3]):
            en = 3 * count3
            max_item = 5
            while en > 0:
                item = rng.randint(0, 2)
                if self.mecha_upgrade[group_idx][item] < max_item:
                    self.mecha_upgrade[group_idx][item] += 1
                    en -= 1

    def _try_invite_people(self, rng: random.Random) -> bool:
        """尝试拉一个人到训练"""
        invite_person = rng.randint(0, 5)
        invite_train = rng.randint(0, 4)

        space = -1
        for idx in range(MAX_PERSON_PER_TRAIN):
            pid = self.person_distribution[invite_train][idx]
            if pid == -1 and space == -1:
                space = idx
            if pid == invite_person:
                return False

        if space == -1:
            return False

        self.person_distribution[invite_train][space] = invite_person
        return True

    def is_card_shining(self, person_idx: int, train_idx: int) -> bool:
        """判断指定卡是否闪彩"""
        if person_idx < 0 or person_idx >= len(self.persons):
            return False
        p = self.persons[person_idx]
        if p.person_type == PersonType.CARD:
            return p.friendship >= 80 and train_idx == p.card_param.card_type
        return False

    # ===== 评分 =====

    def get_skill_score(self) -> float:
        """技能分"""
        rate = self.pt_score_rate * 1.1 if self.is_qie_zhe else self.pt_score_rate
        return rate * self.skill_pt + self.skill_score

    def final_score(self) -> int:
        """最终总分"""
        if self.scoring_mode == SCORING_NORMAL:
            return self._final_score_rank()
        return self._final_score_rank()  # 默认用评价点模式

    def _final_score_rank(self) -> int:
        """评价点计算"""
        # 简化版：属性*系数 + 技能分
        total = 0
        for i in range(5):
            stat = min(self.five_status[i], self.five_status_limit[i])
            # 简化的评分函数
            if stat <= 100:
                total += stat
            else:
                total += int(100 + (stat - 100) * 1.1)
        total += int(self.get_skill_score())
        return total

    def _final_score_sum(self) -> int:
        """属性之和评分"""
        weights = [5, 3, 3, 3, 3]
        total = 0
        for i in range(5):
            real_stat = min(self.five_status[i], self.five_status_limit[i])
            if real_stat > 1200:
                real_stat = 1200 + (real_stat - 1200) / 2
            total += weights[i] * real_stat
        total += self.get_skill_score()
        return int(max(0, total))

    # ===== Link角色判断 =====

    def _is_link_chara_initial_en(self, chara_id: int) -> bool:
        if chara_id > 100000:
            chara_id //= 100
        return chara_id in (1023, 1036)

    def _is_link_chara_more_gear(self, chara_id: int) -> bool:
        if chara_id > 100000:
            chara_id //= 100
        return chara_id in (1023, 1050, 1084)

    def _is_link_chara_initial_overdrive(self, chara_id: int) -> bool:
        if chara_id > 100000:
            chara_id //= 100
        return chara_id in (1050, 1083)

    def _is_link_chara_lv_bonus(self, chara_id: int) -> bool:
        if chara_id > 100000:
            chara_id //= 100
        return chara_id == 1036

    def _is_link_chara_initial_lv(self, chara_id: int) -> bool:
        if chara_id > 100000:
            chara_id //= 100
        return chara_id in (1083, 1084)

    # ===== 输出 =====

    def __repr__(self) -> str:
        status_str = " ".join(f"{v}" for v in self.five_status)
        return (f"Turn={self.turn} Vital={self.vital}/{self.max_vital} "
                f"Mot={self.motivation} [{status_str}] Pt={self.skill_pt}")

    def print_state(self) -> str:
        """格式化输出游戏状态"""
        lines = []
        lines.append(f"=== 第 {self.turn} 回合 ===")
        lines.append(f"体力: {self.vital}/{self.max_vital}  干劲: {self.motivation}")
        names = ["速度", "耐力", "力量", "根性", "智力"]
        for i, name in enumerate(names):
            lines.append(f"  {name}: {self.five_status[i]}/{self.five_status_limit[i]}")
        lines.append(f"技能点: {self.skill_pt}  技能分: {self.skill_score}")
        lines.append(f"训练等级: {self.train_level_count}")
        return "\n".join(lines)
