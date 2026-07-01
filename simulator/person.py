"""
赛马娘AI训练框架 - Person类

参考 UmaAi 的 Person.h，定义支援卡人头信息。
包含支援卡参数、羁绊、闪彩状态等。
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import IntEnum
import random


class PersonType(IntEnum):
    """人头类型枚举"""
    UNKNOWN = 0
    FRIEND_CARD = 1   # 友人支援卡
    CARD = 2          # 普通支援卡
    NPC = 3           # NPC
    YAYOI = 4         # 理事长
    REPORTER = 5      # 记者
    OTHER_FRIEND = 6  # 其他友人卡
    GROUP_CARD = 7    # 团队卡


class FriendStage(IntEnum):
    """友人卡阶段"""
    NOT_CLICKED = 0              # 未点击
    BEFORE_UNLOCK_OUTGOING = 1   # 已点击但未解锁出行
    AFTER_UNLOCK_OUTGOING = 2    # 已解锁出行
    REFUSED_OUTGOING = 3         # 拒绝出行


@dataclass
class SupportCard:
    """支援卡参数
    
    参考 UmaAi 的 SupportCard.h，简化版只保留训练计算所需参数。
    完整的支援卡数据从JSON数据库加载。
    """
    card_id: int = 0              # 支援卡ID
    chara_id: int = 0             # 角色ID
    card_type: int = 0            # 类型(0速1耐2力3根4智5团队6友人)
    card_name: str = ""           # 名称
    
    # 训练相关参数
    you_qing_basic: float = 0.0   # 友情加成
    gan_jing_basic: float = 0.0   # 干劲加成
    xun_lian_basic: float = 0.0   # 训练加成
    bonus_basic: List[float] = field(default_factory=lambda: [0.0] * 6)  # 速耐力根智pt加成
    wiz_vital_bonus: int = 0      # 智力彩圈体力回复
    initial_bonus: List[int] = field(default_factory=lambda: [0] * 6)  # 初期加成
    initial_ji_ban: int = 0       # 初始羁绊
    sai_hou: float = 0.0          # 赛后加成
    
    hint_level: int = 0           # hint等级
    hint_prob_increase: float = 0.0  # hint概率提升
    de_yi_lv: float = 0.0        # 得意率
    fail_rate_drop: float = 0.0   # 失败率下降
    vital_cost_drop: float = 0.0  # 体力消费下降
    
    is_link: bool = False         # 是否为link卡
    
    # 友人卡特有
    event_recovery_amount_up: int = 0  # 友人卡事件体力加成
    event_effect_up: int = 0           # 友人卡事件属性加成
    
    # 固有效果（对齐C++ SupportCard，用于NN输入编码77维布局）
    is_db_card: bool = False           # 是否为数据库卡（有固有效果数据）
    unique_effect_type: int = 0        # 固有效果类型（0=无，1/2=条件型，3-22=其他）
    unique_effect_param: List[float] = field(default_factory=lambda: [0.0] * 6)  # 固有效果参数
    
    # 得意率分布（用于随机分配人头到训练）
    deyilv_distribution: List[float] = field(default_factory=lambda: [100.0] * 5 + [200.0])

    def get_card_effect(self, is_shining: bool, at_train: int, ji_ban: int,
                        effect_factor: int, training_card_num: int,
                        training_shining_num: int) -> "CardTrainingEffect":
        """根据游戏状态计算支援卡的训练效果
        
        Args:
            is_shining: 该卡是否闪彩
            at_train: 所在训练类型
            ji_ban: 当前羁绊值
            effect_factor: 固有效果因子(cardRecord)
            training_card_num: 训练人头数
            training_shining_num: 闪彩人头数
        """
        effect = CardTrainingEffect()
        
        # 训练加成
        effect.xun_lian = self.xun_lian_basic
        effect.gan_jing = self.gan_jing_basic
        
        # 属性加成
        for i in range(6):
            effect.bonus[i] = self.bonus_basic[i]
        
        # 失败率和体力消耗
        effect.fail_rate_drop = self.fail_rate_drop
        effect.vital_cost_drop = self.vital_cost_drop
        
        # 闪彩时才有友情加成
        if is_shining:
            effect.you_qing = self.you_qing_basic
            # 智力彩圈回复体力
            if at_train == 4:  # WIT训练
                effect.vital_bonus = self.wiz_vital_bonus
        
        return effect

    def get_deyilv_train(self, rng: random.Random) -> int:
            """根据得意率随机分配到某个训练
            
            Returns:
                0-4对应速耐力根智，5表示不出现
            """
            total = sum(self.deyilv_distribution)
            r = rng.random() * total
            cumulative = 0.0
            for i, prob in enumerate(self.deyilv_distribution):
                cumulative += prob
                if r < cumulative:
                    return i if i < 5 else 5
            return 5


@dataclass
class CardTrainingEffect:
    """支援卡的训练效果（计算结果）"""
    you_qing: float = 0.0       # 友情加成（闪彩时生效）
    gan_jing: float = 0.0       # 干劲加成
    xun_lian: float = 0.0       # 训练加成
    bonus: List[float] = field(default_factory=lambda: [0.0] * 6)  # 速耐力根智pt加成
    vital_bonus: int = 0        # 体力回复量（智彩圈）
    fail_rate_drop: float = 0.0  # 失败率下降
    vital_cost_drop: float = 0.0  # 体力消费下降


@dataclass
class Person:
    """
    人头类 - 任何一个可能出现在训练里的人
    
    参考 UmaAi 的 Person.h。
    """
    card_param: SupportCard = field(default_factory=SupportCard)
    person_type: int = PersonType.UNKNOWN
    chara_id: int = 0
    friendship: int = 0           # 羁绊值
    is_hint: bool = False         # 是否有hint
    card_record: int = 0          # 随时间变化的参数
    deyilv_distribution: List[float] = field(default_factory=lambda: [100.0] * 5 + [200.0])
    extra_deyilv_bonus: int = 0   # 额外得意率加成
    lianghua_effect: bool = False  # 凉花固有是否生效

    def set_card(self, card_id: int, card_db: Optional[dict] = None):
        """设置为某个支援卡
        
        Args:
            card_id: 支援卡ID
            card_db: 支援卡数据库（可选）
        """
        self.card_param.card_id = card_id
        
        if card_db and str(card_id) in card_db:
            data = card_db[str(card_id)]
            self._load_from_dict(data)
        else:
            # 简单根据ID推断类型
            card_type = (card_id // 10) % 10
            self.card_param.card_type = min(card_type, 6)
            self._setup_deyilv_distribution()

    def _load_from_dict(self, data: dict):
        """从字典加载支援卡数据"""
        cp = self.card_param
        cp.chara_id = data.get("charaId", 0)
        cp.card_type = data.get("cardType", 0)
        cp.card_name = data.get("cardName", "")
        cp.you_qing_basic = data.get("youQingBasic", 0.0)
        cp.gan_jing_basic = data.get("ganJingBasic", 0.0)
        cp.xun_lian_basic = data.get("xunLianBasic", 0.0)
        cp.bonus_basic = data.get("bonusBasic", [0.0] * 6)
        cp.wiz_vital_bonus = data.get("wizVitalBonus", 0)
        cp.initial_bonus = data.get("initialBonus", [0] * 6)
        cp.initial_ji_ban = data.get("initialJiBan", 0)
        cp.sai_hou = data.get("saiHou", 0.0)
        cp.hint_level = data.get("hintLevel", 0)
        cp.hint_prob_increase = data.get("hintProbIncrease", 0.0)
        cp.de_yi_lv = data.get("deYiLv", 0.0)
        cp.fail_rate_drop = data.get("failRateDrop", 0.0)
        cp.vital_cost_drop = data.get("vitalCostDrop", 0.0)
        cp.is_link = data.get("isLink", False)
        cp.event_recovery_amount_up = data.get("eventRecoveryAmountUp", 0)
        cp.event_effect_up = data.get("eventEffectUp", 0)
        
        # 固有效果（对齐C++ SupportCard）
        cp.is_db_card = data.get("isDBCard", False)
        cp.unique_effect_type = data.get("uniqueEffectType", 0)
        cp.unique_effect_param = data.get("uniqueEffectParam", [0.0] * 6)
        
        self.friendship = cp.initial_ji_ban
        self._setup_deyilv_distribution()

    def _setup_deyilv_distribution(self):
        """根据支援卡类型设置得意率分布"""
        cp = self.card_param
        base = [100.0] * 5 + [200.0]  # 速耐力根智 + 不出现
        
        # 根据卡片类型调整分布
        card_type = cp.card_type
        if card_type < 5:  # 普通训练卡
            base[card_type] = 500.0  # 主要出现在对应训练
        
        # 加上得意率加成
        deyilv = cp.de_yi_lv
        if deyilv > 0:
            for i in range(5):
                base[i] += deyilv
        
        # 加上额外得意率加成
        if self.extra_deyilv_bonus > 0:
            for i in range(5):
                base[i] += self.extra_deyilv_bonus
        
        # 凉花固有效果
        if self.lianghua_effect and card_type < 5:
            base[card_type] += 30
        
        self.deyilv_distribution = base
        cp.deyilv_distribution = base

    def set_extra_deyilv_bonus(self, deyilv_bonus: int, lianghua_effect: bool):
        """设置额外得意率加成"""
        self.extra_deyilv_bonus = deyilv_bonus
        self.lianghua_effect = lianghua_effect
        self._setup_deyilv_distribution()

    def get_deyilv_train(self, rng: random.Random) -> int:
        """根据得意率随机分配到某个训练"""
        return self.card_param.get_deyilv_train(rng)

    def is_card_shining(self, train_idx: int) -> bool:
        """判断指定训练是否闪彩
        
        闪彩条件：羁绊 >= 80 且所在训练与卡片类型匹配
        """
        if self.person_type == PersonType.CARD:
            return self.friendship >= 80 and train_idx == self.card_param.card_type
        return False
