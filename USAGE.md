# 赛马娘AI训练框架使用指南

## Colab使用方法

### 1. 打开Colab
1. 浏览器访问 https://colab.research.google.com/
2. 登录Google账号
3. 点击「文件」→「打开笔记本」→「GitHub」标签页
4. 输入仓库地址：`xf8410/uma-train`
5. 选择 `colab/train_uma.ipynb` 打开

### 2. 选择GPU
1. 点击右上角「连接」旁边的下拉箭头
2. 选择「更改运行时类型」
3. 硬件加速器选 **T4 GPU**
4. 点击保存

### 3. 运行训练
按顺序运行每个单元格：
1. **环境检查** - 确认GPU可用
2. **拉取代码** - 把仓库里的GITHUB_REPO改成 `xf8410/uma-train`
3. **安装依赖** - pip install
4. **验证框架** - 确认Game/MCTS/模型都正常
5. **生成训练数据** - 先用5a自我对弈，或5b随机数据测试
6. **开始训练** - 设置参数后运行
7. **推送模型** - 填写GitHub Token后推送到仓库

### 4. 注意事项
- **免费T4 GPU**：每天可用约12小时，断开连接后数据丢失
- **模型大小**：默认配置(1,96,2,192,192)约300KB，满足376KB限制
- **训练数据**：自我对弈生成的数据在Colab关闭后会丢失，重要结果推到GitHub
- **断点续训**：每次保存模型后可以从断点继续训练

---

## Kaggle使用方法

### 1. 创建Notebook
1. 访问 https://www.kaggle.com/
2. 登录账号
3. 点击「Code」→「New Notebook」

### 2. 开启GPU
1. 右侧面板 → Settings → Accelerator
2. 选择 **GPU T4 x2**（或P100）
3. 确认开启

### 3. 添加数据集
1. 右侧「Add Data」→ 搜索 `xf8410/uma-train`
2. 或手动上传train_uma.ipynb的内容

### 4. 运行代码
把Colab笔记本的代码复制到Kaggle的cell里运行，注意：
- Kaggle的路径不同，代码目录在 `/kaggle/working/`
- 先 `!git clone https://github.com/xf8410/uma-train.git`
- 然后 `%cd uma-train`

### 5. 注意事项
- **免费GPU**：每周30小时
- **数据持久化**：Output会自动保存，可以在Kaggle Datasets里复用
- **GitHub Token**：不要硬编码在公开Notebook里，用Kaggle Secrets

---

## AlphaZero训练循环（推荐流程）

```
第0代：手写逻辑自我对弈 → 生成训练数据 → 训练神经网络
第1代：神经网络+手写逻辑自我对弈 → 新训练数据 → 继续训练
第2代：更好的神经网络自我对弈 → ...循环迭代
```

### Colab示例代码

```python
from training.selfplay import SelfPlayWorker
from training.train import train
from model.network import create_model, load_model

for generation in range(10):
    print(f"=== 第{generation}代 ===")
    
    # 1. 自我对弈生成数据
    worker = SelfPlayWorker(model=current_model)
    x, label = worker.generate_batch(50, rng)
    np.savez(f'./data/sp_gen{generation}.npz', x=x, label=label)
    
    # 2. 训练神经网络
    train(f'./data/sp_gen{generation}.npz', max_step=2000)
    
    # 3. 加载新模型
    current_model = load_model('./saved_models/ems/model.pth')
```

---

## 模型转ONNX（给手机App用）

```python
import torch
from model.network import load_model

model = load_model('model.pth')
model.eval()

dummy_input = torch.randn(1, 1121)  # Game_Input_C维度
torch.onnx.export(
    model, dummy_input, "uma_model.onnx",
    input_names=["state"],
    output_names=["policy_value"],
    dynamic_axes={"state": {0: "batch"}, "policy_value": {0: "batch"}}
)
```

---

## 仓库地址
- 代码：https://github.com/xf8410/uma-train
- 数据：https://github.com/xf8410/uma-data
- 插件：https://github.com/xf8410/uma-hook
- App：https://github.com/xf8410/uma-juece
