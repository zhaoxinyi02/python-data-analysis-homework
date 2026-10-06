# Python 数据分析作业 Week 4

## Netflix 数据 Elastic Net 分析

本周使用 Kaggle 常见的 **Netflix Movies and TV Shows** 数据集，对节目元数据进行 Elastic Net 回归分析。由于该数据集的主要数值字段是 `release_year`，本作业以 `release_year` 为响应变量，分析节目类型、分级、首个类别、类别数量、国家数量、标题和简介长度，以及是否存在导演和演员信息等变量。

### 数据处理

- 删除 `release_year` 缺失的记录；
- `type`、`rating` 和 `listed_in` 的首个类别作为分类变量；
- 将国家数、类别数、标题长度、简介长度、导演/演员是否缺失转换为数值特征；
- 使用 `Pipeline`，只在训练集上拟合缺失值填补、one-hot 编码和标准化，避免数据泄漏；
- 训练集与测试集按 80%/20% 划分，随机种子为 42。

### 模型与结果

使用 `ElasticNetCV`，10 折交叉验证，在多个 `alpha` 和 `l1_ratio` 候选值中选择模型。运行代码后，结果保存在 `results/`：

- `netflix_elastic_net_metrics.csv`：最优超参数、测试集 RMSE、MAE、\(R^2\) 与非零特征数；
- `netflix_elastic_net_selected_features.csv`：经过 Elastic Net 保留的非零特征及其系数；
- `netflix_elastic_net_coefficients.png`：绝对系数最大的非零特征图。

本次运行结果如下：

| 指标 | 结果 |
| --- | ---: |
| 样本量 | 6234 |
| 最优 alpha | 0.002656 |
| 最优 l1_ratio | 0.90 |
| 测试集 RMSE | 6.7530 |
| 测试集 MAE | 4.1814 |
| 测试集 \(R^2\) | 0.2856 |
| 非零特征数 | 48 |

### 结论

最优 `l1_ratio=0.90`，说明当前数据更接近 Lasso 的稀疏惩罚，但仍保留了一部分 Ridge 的稳定收缩作用。Elastic Net 将大量 one-hot 特征的系数压缩为 0，只保留 48 个非零特征，起到了变量筛选和控制过拟合的作用。

绝对系数较大的变量主要来自节目类别和内容分级，例如 `Classic Movies`、`TV Shows`、`G` 和 `TV-MA`。这些系数表示在控制其他特征后，与节目发行年份的条件关联，不应直接解释为因果关系。测试集 \(R^2\) 约为 0.286，说明元数据可以解释一部分发行年份差异，但仍有相当部分变化来自数据中未观测的因素。

运行：

```bash
pip install numpy pandas scikit-learn matplotlib
python homework-week4/code/netflix_elastic_net.py
```
