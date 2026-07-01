"""
赛马娘AI训练框架 - 动作定义

参考 UmaAi 的 Action.h，定义所有可能的动作。
包含训练、休息、外出、比赛等动作类型。
"""

from enum import IntEnum
from dataclasses import dataclass


class TrainActionType(IntEnum):
    """训练动作类型枚举"""
    SPEED = 0      # 速度训练
    STAMINA = 1    # 耐力训练
    POWER = 2      # 力量训练
    GUTS = 3       # 根性训练
    WIT = 4        # 智力训练
    REST = 5       # 休息
    OUTGOING = 6   # 外出（含友人出行）
    RACE = 7       # 比赛
    NONE = -1      # 不训练（仅做菜/升级）


# 训练名称映射
TRAINING_NAMES = {
    TrainActionType.SPEED: "速度",
    TrainActionType.STAMINA: "耐力",
    TrainActionType.POWER: "力量",
    TrainActionType.GUTS: "根性",
    TrainActionType.WIT: "智力",
    TrainActionType.REST: "休息",
    TrainActionType.OUTGOING: "外出",
    TrainActionType.RACE: "比赛",
    TrainActionType.NONE: "无",
}

# 游戏阶段
class GameStage(IntEnum):
    """游戏阶段枚举"""
    BEFORE_TRAIN = 1          # 训练（或比赛）前
    BEFORE_MECHA_UPGRADE = 2  # 升级机甲前（Dreams剧本）


@dataclass
class Action:
    """
    动作类
    
    参考UmaAi的Action结构：
    - type: 对应游戏阶段（1=训练回合，2=升级回合）
    - train: 训练类型（-1=不训练，0-4=速耐力根智，5=外出，6=休息，7=比赛）
    - overdrive: 是否开启overdrive
    - mecha_head: 头部升级
    - mecha_chest: 胸部升级
    """
    type: int = 0       # 游戏阶段
    train: int = -1     # 训练类型
    overdrive: bool = False  # 是否开overdrive
    mecha_head: int = 0   # 头部升级数量
    mecha_chest: int = 0  # 胸部升级数量

    # 最大标准动作数（训练8种 + overdrive5种 + 1种不训练 = 14，升级36种）
    MAX_ACTION_TYPE = 14 + 36

    @classmethod
    def from_int(cls, action_id: int) -> "Action":
        """从整数ID构造动作
        
        ID编码规则：
        0-4: 训练(速耐力根智)
        5: 休息
        6: 外出
        7: 比赛
        8-12: 训练+overdrive
        13: 不训练+overdrive
        14-49: 升级动作 (head * 6 + chest，foot = total - head - chest)
        """
        if action_id < 0:
            return cls(type=0, train=-1)
        
        if action_id < 8:
            # 普通训练动作
            return cls(type=GameStage.BEFORE_TRAIN, train=action_id, overdrive=False)
        elif action_id < 13:
            # 训练+overdrive（0-4对应速耐力根智）
            return cls(type=GameStage.BEFORE_TRAIN, train=action_id - 8, overdrive=True)
        elif action_id == 13:
            # 仅开overdrive不训练（C++: overdrive && train==-1 → 5+8=13）
            return cls(type=GameStage.BEFORE_TRAIN, train=-1, overdrive=True)
        elif action_id < 50:
            # 升级动作
            head = (action_id - 14) // 6
            chest = (action_id - 14) % 6
            return cls(type=GameStage.BEFORE_MECHA_UPGRADE, mecha_head=head, mecha_chest=chest)
        else:
            raise ValueError(f"未知的动作ID: {action_id}")

    def to_int(self) -> int:
        """将动作转换为整数ID"""
        if self.type == GameStage.BEFORE_MECHA_UPGRADE:
            return 14 + self.mecha_head * 6 + self.mecha_chest
        elif self.type == GameStage.BEFORE_TRAIN:
            if self.overdrive:
                return self.train + 8 if self.train >= 0 else 13
            else:
                return self.train if self.train >= 0 else -1
        return -1

    def is_standard(self) -> bool:
        """是否为标准动作（可以转换为整数ID）"""
        return self.to_int() >= 0

    def __str__(self) -> str:
        if self.type == GameStage.BEFORE_MECHA_UPGRADE:
            foot = -1  # 需要外部计算
            return f"升级(头={self.mecha_head}, 胸={self.mecha_chest})"
        
        parts = []
        if self.overdrive:
            parts.append("overdrive")
        if self.train >= 0:
            parts.append(TRAINING_NAMES.get(TrainActionType(self.train), f"训练{self.train}"))
        elif self.train == -1:
            parts.append("不训练")
        return "+".join(parts) if parts else "空动作"
