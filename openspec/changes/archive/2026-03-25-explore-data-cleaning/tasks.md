## 1. 项目基础搭建

- [x] 1.1 创建 `explore_data.py` 文件，添加必要的 import 语句 (pandas, numpy, matplotlib, sklearn)
- [x] 1.2 编写 `load_data()` 函数：加载 CSV，打印行数、列名、各标签计数
- [x] 1.3 编写 `report_missing()` 函数：输出每列的缺失值计数和百分比

## 2. 特征清洗

- [x] 2.1 编写 `clean_ordinal()` 函数：从 Likert 字符串 (如 "4 - Agree") 提取数字
- [x] 2.2 编写 `clean_numeric()` 函数：处理 emotion_intensity, num_colours, num_objects 的类型转换和缺失值
- [x] 2.3 编写 `clean_willing_to_pay()` 函数：用正则表达式从自由文本中提取金额
- [x] 2.4 编写 `encode_multiselect()` 函数：拆分逗号分隔的多选列，生成 One-Hot 列
- [x] 2.5 编写 `drop_empty_rows()` 函数：删除全空记录

## 3. 数据标准化与划分

- [x] 3.1 编写 `scale_features()` 函数：对数值特征使用 StandardScaler，保存 mean/std 参数
- [x] 3.2 编写 `split_data()` 函数：80/20 stratified split，random_state=42
- [x] 3.3 保存清洗后的数据集 (CSV 或 npy 格式) 供团队成员使用

## 4. 数据探索与可视化

- [x] 4.1 编写 `plot_class_distribution()` 函数：绘制各画作类别的样本数柱状图
- [x] 4.2 编写 `plot_feature_distributions()` 函数：按画作分组的特征分布直方图
- [x] 4.3 编写 `plot_correlation_matrix()` 函数：数值特征相关性热力图

## 5. 主流程与验证

- [x] 5.1 编写 `main()` 函数串联所有步骤
- [x] 5.2 运行脚本验证输出：确认清洗后的数据形状、无 NaN、图表正常生成
