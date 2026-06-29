"""
赛马娘AI训练框架 - Game类

参考 UmaAi 的 Game.h/cpp，Python版游戏模拟器核心。
重构后：Game dataclass保留字段+基本状态操作，
训练计算委托game_calc、事件处理委托game_events、
Dreams剧本委托game_dreams、评分委托game_score。
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
from .bad_condition import BadConditionType, BadCondition, BadConditionManager
from .formula import FormulaLayer, MOTIVATION_MULTIPLIER, MOTIVATION_REST_VITAL
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
    # バッドコンディション相关常量
    BC_LAZY_TRIGGER_PROB, BC_LAZY_COOLDOWN, BC_LAZY_INITIAL_COOLDOWN,
    BC_SKIN_MOTIVATION_DRAIN_PROB, MOTIVATION_DOWN_COOLDOWN,
    BC_HEAL_REST_LATE_BED, BC_HEAL_REST_SKIN,
    # 新增命名常量
    PSID_NONE, PSID_NONCARD_YAYOI, PSID_NONCARD_REPORTER, PSID_NPC,
    FRIEND_TYPE_NONE, FRIEND_TYPE_LIANGHUA, FRIEND_TYPE_YAYOI,
    PROPERTY_DOUBLE_THRESHOLD, FAIL_RATE_FORMULA_DENOM,
    URA_SKILL_SCORE, INITIAL_SKILL_SCORE_HIGH, INITIAL_SKILL_SCORE_LOW,
    SCORE_ABOVE_100_MULT, SCORING_WEIGHTS, PROPERTY_HALVE_THRESHOLD,
    NONCARD_YAYOI_WEIGHTS, NPC_DISTRIBUTION_WEIGHTS,
    RANDOM_EVENT_PROB, MOTIVATION_DOWN_PROB, CHARA_EVENT_PROB,
    VITAL_EVENT_SMALL_PROB, VITAL_EVENT_BIG_PROB, MOTIVATION_UP_EVENT_PROB,
    HINT_BASE_PROB, OVERDRIVE_INVITE_MAX_TRY,
    ZHONGMA_BLUE_LIMIT_FACTOR, ZHONGMA_BLUE_LIMIT_MULT,
    MECHA_RIVAL_LV_LIMITS, MECHA_GEAR_BONUS_TABLE,
    OVERDRIVE_TRAIN_MULT, OVERDRIVE_EN_COST,
    URA_START_TURN, MECHA_LV_GAIN_BONUS_TABLE,
)

# 导入拆分模块
from . import game_calc
from . import game_events
from . import game_dreams
from . import game_score


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
    bc_manager: BadConditionManager = field(default_factory=BadConditionManager)  # バッドコンディション
    formula: FormulaLayer = field(default=None)  # 公式层（__post_init__中初始化）
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

    def __post_init__(self):
        """初始化公式层"""
        if self.bc_manager is None:
            self.bc_manager = BadConditionManager()
        if self.formula is None:
            self.formula = FormulaLayer(self.bc_manager)

    def _ensure_initialized(self):
        """P2-4修复：确保formula/bc_manager已初始化，防止部分初始化时访问None"""
        if self.formula is None or self.bc_manager is None:
            self.__post_init__()

    def save_snapshot(self) -> dict:
        """保存当前状态快照，比deepcopy快10-50倍

        数值/枚举字段直接复制，一维列表浅拷贝，
        二维列表逐行浅拷贝，复杂对象(persons/bc_manager/formula)用deepcopy。
        """
        return {
            # 参数设置
            "pt_score_rate": self.pt_score_rate,
            "hint_pt_rate": self.hint_pt_rate,
            "event_strength": self.event_strength,
            "scoring_mode": self.scoring_mode,
            # 基本状态
            "uma_id": self.uma_id,
            "is_link_uma": self.is_link_uma,
            "is_racing_turn": list(self.is_racing_turn),
            "five_status_bonus": list(self.five_status_bonus),
            "turn": self.turn,
            "game_stage": self.game_stage,
            "vital": self.vital,
            "max_vital": self.max_vital,
            "motivation": self.motivation,
            "five_status": list(self.five_status),
            "five_status_limit": list(self.five_status_limit),
            "skill_pt": self.skill_pt,
            "skill_score": self.skill_score,
            "train_level_count": list(self.train_level_count),
            "failure_rate_bias": self.failure_rate_bias,
            "bc_manager": deepcopy(self.bc_manager),
            "formula": deepcopy(self.formula),
            "is_qie_zhe": self.is_qie_zhe,
            "is_ai_jiao": self.is_ai_jiao,
            "is_positive_thinking": self.is_positive_thinking,
            "is_refresh_mind": self.is_refresh_mind,
            # 种马
            "zhong_ma_blue_count": list(self.zhong_ma_blue_count),
            "zhong_ma_extra_bonus": list(self.zhong_ma_extra_bonus),
            "is_racing": self.is_racing,
            # 非卡羁绊
            "friendship_noncard_yayoi": self.friendship_noncard_yayoi,
            "friendship_noncard_reporter": self.friendship_noncard_reporter,
            # 人头
            "persons": deepcopy(self.persons),
            "person_distribution": [list(row) for row in self.person_distribution],
            # 赛后加成
            "saihou": self.saihou,
            # 友人卡
            "friend_type": self.friend_type,
            "friend_is_ssr": self.friend_is_ssr,
            "friend_person_id": self.friend_person_id,
            "friend_stage": self.friend_stage,
            "friend_outgoing_used": self.friend_outgoing_used,
            "friend_vital_bonus": self.friend_vital_bonus,
            "friend_status_bonus": self.friend_status_bonus,
            # 得意率
            "current_deyilv_bonus": self.current_deyilv_bonus,
            "current_lianghua_effect_enable": self.current_lianghua_effect_enable,
            # Dreams剧本相关
            "mecha_linkeffect_gear_prob_bonus": self.mecha_linkeffect_gear_prob_bonus,
            "mecha_linkeffect_lvbonus": self.mecha_linkeffect_lvbonus,
            "mecha_rival_lv": list(self.mecha_rival_lv),
            "mecha_overdrive_energy": self.mecha_overdrive_energy,
            "mecha_overdrive_enabled": self.mecha_overdrive_enabled,
            "mecha_en": self.mecha_en,
            "mecha_upgrade": [list(row) for row in self.mecha_upgrade],
            "mecha_has_gear": list(self.mecha_has_gear),
            "mecha_win_history": list(self.mecha_win_history),
            "mecha_any_lose": self.mecha_any_lose,
            # 可计算的非独立信息
            "train_value": [list(row) for row in self.train_value],
            "train_vital_change": list(self.train_vital_change),
            "fail_rate": list(self.fail_rate),
            "is_train_shining": list(self.is_train_shining),
            "train_value_lower": [list(row) for row in self.train_value_lower],
            # 中间变量
            "mecha_rival_lv_total": self.mecha_rival_lv_total,
            "mecha_rival_lv_limit": self.mecha_rival_lv_limit,
            "mecha_upgrade_total": list(self.mecha_upgrade_total),
            "mecha_lv_gain": [list(row) for row in self.mecha_lv_gain],
            "mecha_training_status_multiplier": list(self.mecha_training_status_multiplier),
            "mecha_lv_gain_multiplier": list(self.mecha_lv_gain_multiplier),
        }

    def restore_snapshot(self, snap: dict):
        """从快照恢复状态，与save_snapshot逐字段对应

        注意：所有可变字段（列表/对象）都必须创建新副本，
        否则游戏运行时的原地修改会破坏快照dict。
        """
        # 参数设置（不变参数可直接赋值）
        self.pt_score_rate = snap["pt_score_rate"]
        self.hint_pt_rate = snap["hint_pt_rate"]
        self.event_strength = snap["event_strength"]
        self.scoring_mode = snap["scoring_mode"]
        # 基本状态
        self.uma_id = snap["uma_id"]
        self.is_link_uma = snap["is_link_uma"]
        self.is_racing_turn = list(snap["is_racing_turn"])
        self.five_status_bonus = list(snap["five_status_bonus"])
        self.turn = snap["turn"]
        self.game_stage = snap["game_stage"]
        self.vital = snap["vital"]
        self.max_vital = snap["max_vital"]
        self.motivation = snap["motivation"]
        self.five_status = list(snap["five_status"])
        self.five_status_limit = list(snap["five_status_limit"])
        self.skill_pt = snap["skill_pt"]
        self.skill_score = snap["skill_score"]
        self.train_level_count = list(snap["train_level_count"])
        self.failure_rate_bias = snap["failure_rate_bias"]
        self.bc_manager = deepcopy(snap["bc_manager"])
        self.formula = deepcopy(snap["formula"])
        self.is_qie_zhe = snap["is_qie_zhe"]
        self.is_ai_jiao = snap["is_ai_jiao"]
        self.is_positive_thinking = snap["is_positive_thinking"]
        self.is_refresh_mind = snap["is_refresh_mind"]
        # 种马
        self.zhong_ma_blue_count = list(snap["zhong_ma_blue_count"])
        self.zhong_ma_extra_bonus = list(snap["zhong_ma_extra_bonus"])
        self.is_racing = snap["is_racing"]
        # 非卡羁绊
        self.friendship_noncard_yayoi = snap["friendship_noncard_yayoi"]
        self.friendship_noncard_reporter = snap["friendship_noncard_reporter"]
        # 人头
        self.persons = deepcopy(snap["persons"])
        self.person_distribution = [list(row) for row in snap["person_distribution"]]
        # 赛后加成
        self.saihou = snap["saihou"]
        # 友人卡
        self.friend_type = snap["friend_type"]
        self.friend_is_ssr = snap["friend_is_ssr"]
        self.friend_person_id = snap["friend_person_id"]
        self.friend_stage = snap["friend_stage"]
        self.friend_outgoing_used = snap["friend_outgoing_used"]
        self.friend_vital_bonus = snap["friend_vital_bonus"]
        self.friend_status_bonus = snap["friend_status_bonus"]
        # 得意率
        self.current_deyilv_bonus = snap["current_deyilv_bonus"]
        self.current_lianghua_effect_enable = snap["current_lianghua_effect_enable"]
        # Dreams剧本相关
        self.mecha_linkeffect_gear_prob_bonus = snap["mecha_linkeffect_gear_prob_bonus"]
        self.mecha_linkeffect_lvbonus = snap["mecha_linkeffect_lvbonus"]
        self.mecha_rival_lv = list(snap["mecha_rival_lv"])
        self.mecha_overdrive_energy = snap["mecha_overdrive_energy"]
        self.mecha_overdrive_enabled = snap["mecha_overdrive_enabled"]
        self.mecha_en = snap["mecha_en"]
        self.mecha_upgrade = [list(row) for row in snap["mecha_upgrade"]]
        self.mecha_has_gear = list(snap["mecha_has_gear"])
        self.mecha_win_history = list(snap["mecha_win_history"])
        self.mecha_any_lose = snap["mecha_any_lose"]
        # 可计算的非独立信息
        self.train_value = [list(row) for row in snap["train_value"]]
        self.train_vital_change = list(snap["train_vital_change"])
        self.fail_rate = list(snap["fail_rate"])
        self.is_train_shining = list(snap["is_train_shining"])
        self.train_value_lower = [list(row) for row in snap["train_value_lower"]]
        # 中间变量
        self.mecha_rival_lv_total = snap["mecha_rival_lv_total"]
        self.mecha_rival_lv_limit = snap["mecha_rival_lv_limit"]
        self.mecha_upgrade_total = list(snap["mecha_upgrade_total"])
        self.mecha_lv_gain = [list(row) for row in snap["mecha_lv_gain"]]
        self.mecha_training_status_multiplier = list(snap["mecha_training_status_multiplier"])
        self.mecha_lv_gain_multiplier = list(snap["mecha_lv_gain_multiplier"])

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
        # 初始技能分公式系数（3星以上: 170*(stars-2), 以下: 120*stars）
        self.skill_score = INITIAL_SKILL_SCORE_HIGH * (uma_stars - 2) if uma_stars >= 3 else INITIAL_SKILL_SCORE_LOW * uma_stars
        self.failure_rate_bias = 0
        self.bc_manager = BadConditionManager()
        self.formula = FormulaLayer(self.bc_manager)
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

        # 属性上限和初始值加成（种马蓝因子属性上限加成系数）
        for i in range(5):
            self.five_status_limit[i] += int(self.zhong_ma_blue_count[i] * ZHONGMA_BLUE_LIMIT_FACTOR * ZHONGMA_BLUE_LIMIT_MULT)
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
        return game_calc.calc_real_status_gain(self, value, gain)

    def add_status(self, idx: int, value: int):
        """增加属性值，处理上限和1200翻倍"""
        t = self.five_status[idx] + value
        t = min(t, self.five_status_limit[idx])
        t = max(t, 1)
        if t > PROPERTY_DOUBLE_THRESHOLD:
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
        """增加或减少心情，考虑正向思考和片頭痛限制"""
        self._ensure_initialized()
        if value < 0:
            if self.is_positive_thinking:
                self.is_positive_thinking = False
            else:
                self.motivation = max(1, self.motivation + value)
        else:
            # 片頭痛时やる気不上升
            if self.bc_manager.is_motivation_blocked():
                return
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

    # ===== 训练计算（委托game_calc） =====

    def calculate_failure_rate(self, train_type: int, fail_rate_multiply: float = 1.0) -> int:
        """计算训练失败率"""
        return game_calc.calc_failure_rate(self, train_type, fail_rate_multiply)

    def calculate_training_value_single(self, tra: int):
        """计算单个训练的数值"""
        return game_calc.calc_training_value_single(self, tra)

    def calculate_lv_gain_single(self, tra: int, head_num: int, is_shining: bool):
        """计算每个训练加多少研究等级"""
        return game_calc.calc_lv_gain_single(self, tra, head_num, is_shining)

    def calculate_training_value(self):
        """计算所有训练分别加多少"""
        return game_calc.calc_training_value(self)

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

        # 是否有hint（人头hint基础概率）
        for pid in range(6):
            if self.persons[pid].person_type == PersonType.CARD:
                hint_prob = HINT_BASE_PROB * (1 + 0.01 * self.persons[pid].card_param.hint_prob_increase)
                hint_prob *= (1.0 + 0.15 * self.mecha_upgrade[0][1])
                self.persons[pid].is_hint = rng.random() < hint_prob

        # 随机决定是否有齿轮
        gear_prob = MECHA_GEAR_PROB + MECHA_GEAR_PROB_LINK_BONUS * self.mecha_linkeffect_gear_prob_bonus
        for i in range(5):
            self.mecha_has_gear[i] = rng.random() < gear_prob

        # URA期间自动开启overdrive
        if self.turn >= URA_START_TURN and not self.mecha_any_lose:
            self.mecha_overdrive_energy = OVERDRIVE_EN_COST
            self._mecha_activate_overdrive(rng)
        elif overdrive_was_enabled:
            self._mecha_activate_overdrive(rng)

        self.calculate_training_value()

    def _noncard_distribution(self, rng: random.Random) -> int:
        """非卡理事长/记者的分布"""
        return self._weighted_choice(rng, NONCARD_YAYOI_WEIGHTS)

    def _npc_distribution(self, rng: random.Random) -> int:
        """NPC的分布"""
        return self._weighted_choice(rng, NPC_DISTRIBUTION_WEIGHTS)

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
                if self.mecha_overdrive_energy < OVERDRIVE_EN_COST:
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
        self._ensure_initialized()
        if train == TrainActionType.REST:
            # お休み：公式层计算回复量
            great_prob = self.formula.great_success_prob(self.vital, self.max_vital)
            is_great = rng.random() < great_prob
            rest_vital = self.formula.rest_vital(self.motivation, is_great)
            self.add_vital(rest_vital)
            if is_great and not self.bc_manager.is_motivation_blocked():
                self.add_motivation(1)
            elif rng.random() < 0.04 and not self.bc_manager.is_motivation_blocked():
                self.add_motivation(1)
            # お休み概率治愈バッドコンディション
            healed = self.formula.bc_heal_rest(rng)
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
            # なまけ癖跳过训练检查
            if self.bc_manager.should_skip_training(self.turn, rng):
                # なまけ癖触发，跳过本回合训练
                return False
            # 训练
            fail_rate = self.fail_rate[train]
            roll = rng.randint(1, 100)
            
            if roll <= fail_rate:
                # 训练失败
                if fail_rate >= 20:
                    # 大失败
                    for i in range(5):
                        self.add_status(i, -4)
                        if self.five_status[i] > PROPERTY_DOUBLE_THRESHOLD:
                            self.add_status(i, -4)
                    self.add_motivation(-3)
                    self.add_vital(10)
                else:
                    # 小失败
                    self.add_status(train, -5)
                    if self.five_status[train] > PROPERTY_DOUBLE_THRESHOLD:
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

    # ===== 事件处理（委托game_events） =====

    def _check_random_events(self, rng: random.Random):
        """模拟随机事件"""
        return game_events.check_random_events(self, rng)

    def _check_fixed_events(self, rng: random.Random):
        """每回合的固定事件"""
        return game_events.check_fixed_events(self, rng)

    def _check_event_after_train(self, rng: random.Random):
        """训练后检查事件并推进回合"""
        return game_events.check_event_after_train(self, rng)

    def _maybe_update_deyilv(self):
        """更新得意率"""
        return game_events.maybe_update_deyilv(self)

    def _handle_friend_click_event(self, rng: random.Random, at_train: int):
        """友人点击事件"""
        return game_events.handle_friend_click_event(self, rng, at_train)

    def _handle_friend_outgoing(self, rng: random.Random):
        """友人外出"""
        return game_events.handle_friend_outgoing(self, rng)

    def _handle_friend_unlock(self, rng: random.Random):
        """友人外出解锁"""
        return game_events.handle_friend_unlock(self, rng)

    # ===== Dreams剧本相关（委托game_dreams） =====

    def _mecha_maybe_run_uge(self) -> bool:
        """检查是否触发UGE比赛"""
        return game_dreams.mecha_maybe_run_uge(self)

    def _mecha_activate_overdrive(self, rng: random.Random) -> bool:
        """开启overdrive"""
        return game_dreams.mecha_activate_overdrive(self, rng)

    def _mecha_maybe_reverse_overdrive(self) -> bool:
        """恢复overdrive状态"""
        return game_dreams.mecha_maybe_reverse_overdrive(self)

    def _mecha_distribute_en(self, head3: int, chest3: int, foot3: int, rng: random.Random):
        """分配EN到头胸脚升级"""
        return game_dreams.mecha_distribute_en(self, head3, chest3, foot3, rng)

    def _try_invite_people(self, rng: random.Random) -> bool:
        """尝试拉一个人到训练"""
        return game_dreams.try_invite_people(self, rng)

    def is_card_shining(self, person_idx: int, train_idx: int) -> bool:
        """判断指定卡是否闪彩"""
        return game_dreams.is_card_shining(self, person_idx, train_idx)

    def _init_mecha(self, rng: random.Random):
        """初始化Dreams剧本相关状态"""
        return game_dreams.init_mecha(self, rng)

    def _is_link_chara_initial_en(self, chara_id: int) -> bool:
        return game_dreams.is_link_chara_initial_en(self, chara_id)

    def _is_link_chara_more_gear(self, chara_id: int) -> bool:
        return game_dreams.is_link_chara_more_gear(self, chara_id)

    def _is_link_chara_initial_overdrive(self, chara_id: int) -> bool:
        return game_dreams.is_link_chara_initial_overdrive(self, chara_id)

    def _is_link_chara_lv_bonus(self, chara_id: int) -> bool:
        return game_dreams.is_link_chara_lv_bonus(self, chara_id)

    def _is_link_chara_initial_lv(self, chara_id: int) -> bool:
        return game_dreams.is_link_chara_initial_lv(self, chara_id)

    # ===== 评分（委托game_score） =====

    def get_skill_score(self) -> float:
        """技能分"""
        return game_score.calc_skill_score(self)

    def final_score(self) -> int:
        """最终总分"""
        return game_score.calc_final_score(self)

    def _final_score_rank(self) -> int:
        """评价点计算"""
        return game_score._calc_final_score_rank(self)

    def _final_score_sum(self) -> int:
        """属性之和评分"""
        return game_score._calc_final_score_sum(self)

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
        if self.bc_manager.count > 0:
            lines.append(f"バッドコンディション: {self.bc_manager}")
        return "\n".join(lines)
