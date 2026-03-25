## Why

数据清洗流水线 (`explore_data.py`) 已完成，产出了 22 维特征的 train/val 数据集。作为团队中负责逻辑回归 (LR) 的成员，现在需要一个训练脚本来：用 sklearn 训练和调参 LR 模型，生成报告所需的实验数据（超参数对比表和图表），并导出模型参数供最终 `pred.py` 使用。截止日期 3月30日，剩余 5 天。

## What Changes

- **新增** `train_lr.py`: 逻辑回归训练与超参数调优脚本
  - 加载 `cleaned_data/` 中的标准化数据
  - 使用 sklearn LogisticRegression 训练多分类模型 (multinomial + softmax)
  - K-Fold 交叉验证调优正则化参数 C 和 penalty (L1/L2)
  - 生成超参数对比表 (CSV) 和调优曲线图
  - 导出最佳模型的 coef_ 和 intercept_ 为 npy 文件
  - 输出分类报告 (precision/recall/f1) 和混淆矩阵
- **新增** `pred_lr.py`: 纯 numpy 的 LR 推理模块
  - 加载导出的权重 + scaler 参数
  - 实现手动 softmax 前向传播
  - 作为最终 `pred.py` 的候选模块

## Capabilities

### New Capabilities
- `lr-training`: 逻辑回归模型的训练、超参数搜索、评估与参数导出
- `lr-inference`: 纯 numpy 实现的逻辑回归推理（softmax 前向传播）

### Modified Capabilities
_(无)_

## Impact

- **新文件**: `train_lr.py`, `pred_lr.py`
- **输出产物**: `lr_results/` 目录（权重 npy、超参数 CSV、图表 PNG）
- **依赖**: `cleaned_data/` 中的 npy 文件（由 explore_data.py 产出）
- **团队影响**: 调优结果将用于报告 Model Choice & Hyperparameters 部分 (3分)
