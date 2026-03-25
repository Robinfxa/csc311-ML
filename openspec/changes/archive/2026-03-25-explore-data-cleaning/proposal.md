## Why

CSC311 ML Challenge 的截止日期是 3月30日。我们团队需要一个数据探索和清洗脚本来处理 `ml_challenge_dataset.csv`（1832条记录，3幅画各562条）。这是所有后续建模工作（逻辑回归、决策树、朴素贝叶斯）的共同基础，必须首先完成。

## What Changes

- **新增** `explore_data.py`: 数据探索与清洗脚本
  - 加载 CSV 数据并分析缺失值分布
  - 处理 14 个特征列的类型转换与清洗
  - 对数值特征进行标准化 (StandardScaler)
  - 对多选分类特征 (room, view_with, season) 进行 One-Hot 编码
  - 从文本中提取 `willing_to_pay` 的数值
  - 生成 EDA 可视化图表（特征分布、类别相关性）
  - 输出清洗后的数据集供团队成员使用
  - 使用 train/validation split 划分数据集

## Capabilities

### New Capabilities
- `data-cleaning`: 原始 CSV 数据的清洗、类型转换、缺失值处理和特征工程
- `data-exploration`: 特征分布可视化、类别相关性分析和 EDA 图表生成

### Modified Capabilities
_(无已有 capability 需要修改)_

## Impact

- **新文件**: `explore_data.py`
- **依赖库**: `pandas`, `numpy`, `matplotlib`, `sklearn` (仅用于训练阶段的 StandardScaler 等工具)
- **输出产物**: 清洗后的数据集文件 + EDA 可视化图表
- **团队影响**: 产出的清洗数据将作为决策树和朴素贝叶斯同学的输入数据源
