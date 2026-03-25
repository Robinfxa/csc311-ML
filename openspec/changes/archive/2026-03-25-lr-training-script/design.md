## Context

`explore_data.py` 已产出 22 维特征的标准化数据：
- `cleaned_data/X_train.npy` (1300×22), `X_val.npy` (326×22)
- `cleaned_data/y_train.npy`, `y_val.npy` (标签: 3 种画作名称字符串)
- `cleaned_data/scaler_mean.npy`, `scaler_std.npy` (StandardScaler 参数)
- `cleaned_data/feature_names.txt` (22 个特征名)

标签需要编码为整数才能用于 sklearn。最终 `pred.py` 只允许 numpy/pandas。

## Goals / Non-Goals

**Goals:**
- 用 sklearn 训练逻辑回归并系统性调参
- 产出报告所需的所有实验数据和图表
- 导出最佳模型参数，供手搓 numpy 推理使用
- 编写纯 numpy 的 `pred_lr.py` 推理模块，验证与 sklearn 预测一致

**Non-Goals:**
- 不训练决策树或朴素贝叶斯（其他组员负责）
- 不编写最终合并版 `pred.py`（等三个模型对比后再决定）
- 不做文本特征的 NLP 处理

## Decisions

### Decision 1: 多分类策略 → Multinomial + Softmax

**选择**: `LogisticRegression(multi_class='multinomial', solver='lbfgs')`

**理由**: Multinomial 直接优化 softmax 交叉熵损失，比 OVR (One-vs-Rest) 更适合互斥类别。导出参数后用 numpy 写 softmax 也更简洁：`np.argmax(X @ W.T + b)`。

**替代方案**: OVR → 需要训练 3 个独立模型，参数管理更复杂。

### Decision 2: 超参数搜索空间

**选择**: Grid search over:
- `C`: [0.001, 0.01, 0.1, 1, 10, 100]（正则化强度倒数）
- `penalty`: ['l1', 'l2']（L1 需要 solver='saga'）
- 使用 5-Fold Stratified CV

**理由**: C 的搜索范围覆盖 5 个数量级，足以找到最优正则化强度。报告要求展示调参过程，所以需要记录每个超参数组合的表现。

### Decision 3: 标签编码映射

**选择**: 按字母序映射标签 → 整数：
- `The Persistence of Memory` → 0
- `The Starry Night` → 1
- `The Water Lily Pond` → 2

**理由**: 固定映射确保训练和推理阶段一致。保存映射关系到文件。

### Decision 4: Numpy 推理验证

**选择**: 训练后立即用手搓的 numpy softmax 对验证集做预测，与 sklearn `model.predict()` 逐样本对比，确保 100% 一致。

**理由**: 如果 numpy 推理结果与 sklearn 不一致，说明导出或前向传播有 bug。必须在交付前验证。

## Risks / Trade-offs

- **L1 penalty 需要换 solver** → solver='saga' 支持 L1+multinomial，但可能收敛更慢，增加 `max_iter=5000`
- **C 过大导致过拟合** → 通过 CV 分数检测，同时记录 train vs val 差距
- **Numpy 精度与 sklearn 微小差异** → 用 `np.allclose` 而不是严格相等来验证
