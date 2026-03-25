## Context

CSC311 ML Challenge 数据集 (`ml_challenge_dataset.csv`) 包含 1832 条记录，每条对应一位学生对三幅画之一的问卷回答。数据集有 14 个特征列，类型混杂（数值、有序分类、多选、自由文本），且存在缺失值和格式不一致的问题。

当前状态：原始 CSV 文件未经任何处理，团队尚未开始建模工作。

约束：
- 最终 `pred.py` 只允许使用 `numpy`, `pandas`, `sys`, `csv`, `random`
- 探索阶段可使用 `sklearn`, `matplotlib` 等任意库
- 三幅画的标签分布完全均衡 (各 562 条)

## Goals / Non-Goals

**Goals:**
- 产出一个可复用的数据清洗流水线，供团队三位建模同学共用
- 识别并处理所有缺失值和异常值
- 对不同类型特征采用合适的编码方式
- 生成 EDA 可视化，为报告的 Data 部分提供素材
- 输出清洗后的 `X_train`, `X_val`, `y_train`, `y_val` 数据集

**Non-Goals:**
- 不在此脚本中训练任何模型（模型训练是下一步）
- 不处理自由文本特征的 NLP 分析（`describe_feeling`, `food`, `soundtrack`）
- 不编写最终 `pred.py` 推理代码

## Decisions

### Decision 1: 特征分层策略

**选择**: 将特征分为三层逐步加入

| 层级 | 特征 | 处理方式 |
|------|------|----------|
| 一线 (7个) | emotion_intensity, sombre, content, calm, uneasy, num_colours, num_objects | 直接当数值, StandardScaler |
| 二线 (3组→14维) | room(5), view_with(5), season(4) | 拆逗号 → 多标签 One-Hot |
| 三线 (1个) | willing_to_pay | 正则提取数字, log 变换 |

**理由**: 一线特征已经是干净的数值，优先保证 baseline 可用；二线特征需要拆分多选项；三线特征文本格式极其混乱（"$50"、"100,000"、"I wouldn't pay"），清洗成本高但信息量大。

**替代方案**: 直接丢弃 willing_to_pay → 损失一个可能有区分力的特征。

### Decision 2: 有序分类特征的处理

**选择**: 将 Likert 量表 (1-5) 直接提取数字作为数值特征。

**理由**: `"4 - Agree"` → 提取 `4`。这些特征本身有自然顺序，逻辑回归可以直接利用数值大小关系。One-Hot 编码反而会丢失顺序信息。

### Decision 3: 缺失值处理

**选择**: 数值特征用中位数填充，分类特征用众数填充。整行全空的记录直接丢弃。

**理由**: 数据集中存在少量完全空白的回答（仅有 unique_id 和 Painting），这些无法提供任何信息。中位数对异常值比均值更鲁棒。

### Decision 4: 数据划分策略

**选择**: 80/20 train/validation split，使用 stratified split 保持类别比例。

**理由**: 三类平衡，stratified split 确保验证集中各类占比一致。保留固定 random_state 以便实验可复现。

## Risks / Trade-offs

- **willing_to_pay 清洗失败率高** → 对无法解析的值视为 NaN，用中位数填充。记录清洗成功率。
- **多选列可能有意外格式** → 预先检查所有唯一值组合，建立白名单。
- **StandardScaler 的 fit 参数需要保存** → 后续 `pred.py` 推理时需要同样的 mean/std，需额外保存 scaler 参数。
