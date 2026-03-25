## 1. 数据加载与标签编码

- [x] 1.1 创建 `train_lr.py`，加载 `cleaned_data/` 中的 npy 文件
- [x] 1.2 实现标签编码：字符串标签 → 整数 {0,1,2}，保存 `label_map.txt`

## 2. 超参数网格搜索

- [x] 2.1 实现 5-Fold Stratified CV 的网格搜索 (C × penalty)
- [x] 2.2 记录每组超参数的 mean_cv_accuracy, std_cv_accuracy, train_accuracy
- [x] 2.3 保存结果为 `lr_results/hyperparameter_results.csv`
- [x] 2.4 生成调优曲线图 (C vs accuracy, 按 penalty 分线) 保存为 PNG

## 3. 最佳模型训练与评估

- [x] 3.1 用最佳超参数在全部训练数据上重新训练
- [x] 3.2 在验证集上生成分类报告 (precision/recall/F1)
- [x] 3.3 生成混淆矩阵热力图保存为 PNG

## 4. 参数导出

- [x] 4.1 导出 `lr_weights.npy` 和 `lr_bias.npy` 到 `lr_results/`

## 5. 纯 Numpy 推理模块

- [x] 5.1 创建 `pred_lr.py`，实现 softmax 前向传播推理
- [x] 5.2 实现 `predict_all(filename)` 函数：读取原始 CSV → 预处理 → 推理 → 返回画作名列表
- [x] 5.3 验证 numpy 推理结果与 sklearn predict 完全一致

## 6. 运行验证

- [x] 6.1 运行 `train_lr.py` 完成训练，验证所有输出文件生成
- [x] 6.2 运行 `pred_lr.py` 对训练集 CSV 做预测，验证准确率与 sklearn 一致
