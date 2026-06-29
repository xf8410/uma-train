"""
赛马娘AI训练框架 - 全局配置（向后兼容入口）

原config.py已按关注点拆分为4个模块，本文件通过import * re-export所有名字，
保持向后兼容，其他文件无需修改import路径。

拆分模块：
- config_game.py — 游戏常量、BC常量、MECHA常量、人员ID/友人卡类型、训练基本值、评分权重、魔法数字
- config_nn.py — NN维度、模型架构参数
- config_train.py — 训练超参(lr/batch/epoch/stability等)、自我对弈参数、rollback参数
- config_mcts.py — MCTS搜索参数、手写评估参数、NN校准开关
"""

from config_game import *
from config_nn import *
from config_train import *
from config_mcts import *
