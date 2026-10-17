# Python 数据分析课程作业

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Course-Python%20Data%20Analysis-4C8BF5" alt="Course">
  <img src="https://img.shields.io/badge/Semester-2026%20Fall-success" alt="Semester">
  <img src="https://img.shields.io/badge/Status-In%20Progress-orange" alt="Status">
</p>

<p align="center">
  《Python 数据分析》课程作业仓库，用于整理每周理论作业、Python 实验代码与数据分析结果。
</p>

---

## 基本信息

- **学生**：赵鑫亿
- **学期**：2026 秋季学期
- **课程**：Python 数据分析
- **仓库用途**：按周整理并提交课程作业，方便查看、运行与后续复习

## 作业进度

| 周次 | 目录 | 主要内容 | 状态 |
| --- | --- | --- | --- |
| Week 1 | [homework-week1](./homework-week1) | Python 基础：输出 `helloworld` | 已完成 |
| Week 2 | [homework-week2](./homework-week2) | OLS 理论证明、调整后 \(R^2\)、Carseats 多元线性回归与 VIF | 已完成 |
| Week 3 | [homework-week3](./homework-week3) | Ridge、Lasso、Elastic Net 理论与 Hitters 实战 | 已完成 |
| Week 4 | [homework-week4](./homework-week4) | Netflix 数据 Elastic Net 分析 | 已完成 |
| Week 5 | [homework-week5](./homework-week5) | German Credit 数据 SVM 分类（RBF 核 + 类别权重） | 已完成 |

## 仓库结构

```text
python-data-analysis-homework/
├── README.md
├── homework-week1/
│   └── hello.py
└── homework-week2/
    ├── theory.md
    └── code/
        └── carseats_regression.py
├── homework-week3/
│   ├── theory.md
│   └── code/
│       └── regularized_regression_comparison.py
└── homework-week4/
    ├── analysis.md
    └── code/
        └── netflix_elastic_net.py
└── homework-week5/
    ├── analysis.md
    ├── code/
    │   └── credit_svm.py
    └── results/
```

## Week 1

第一周用于验证 Python 基础运行环境。

运行：

```bash
python homework-week1/hello.py
```

输出：

```text
helloworld
```

## Week 2

第二周作业包含理论题与 Python 实践题两部分。

### 理论部分

文件：[homework-week2/theory.md](./homework-week2/theory.md)

主要内容：

1. 在线性回归中证明 OLS 残差满足
   \[
   \sum_{i=1}^{n} e_i = 0
   \]
   以及
   \[
   \sum_{i=1}^{n} x_i e_i = 0
   \]
2. 从 **RSS** 与 **自由度** 的角度解释：
   - 为什么加入无关特征后普通 \(R^2\) 只会上升或保持不变；
   - 为什么调整后的 \(R^2_{adj}\) 可能下降。

### Python 实践部分

文件：[homework-week2/code/carseats_regression.py](./homework-week2/code/carseats_regression.py)

使用 **Carseats** 数据集，以 `Sales` 为响应变量，选取：

- `Price`
- `Income`
- `Advertising`
- `ShelveLoc`

建立多元线性回归模型。

程序会自动完成：

- 加载 Carseats 数据集；
- 建立 OLS 多元线性回归模型；
- 输出完整模型拟合报告；
- 判断 `ShelveLoc` 的基准组；
- 解释 `ShelveLoc[Good]` 的实际商业含义；
- 计算各解释变量的 VIF；
- 判断是否存在明显多重共线性风险。

当前数据下的关键结果约为：

| 指标 | 结果 |
| --- | ---: |
| \(R^2\) | 0.627 |
| ShelveLoc 基准组 | Bad |
| ShelveLoc[Good] 系数 | 4.836 |
| 最大 VIF | 约 1.50 |
| 多重共线性 | 无明显风险 |

其中 `ShelveLoc[Good]` 系数约为 **4.836**，表示在其他解释变量保持不变时，货架位置为 Good 的商店相较于 Bad 的商店，模型预测销量平均高约 **4.836 千件**。

## 运行 Week 2 代码

建议使用 Python 3.x。

安装依赖：

```bash
pip install pandas statsmodels
```

运行：

```bash
python homework-week2/code/carseats_regression.py
```

程序会直接在终端输出模型摘要、系数解释和 VIF 检验结果。

## Week 5：German Credit 数据的 SVM 分类

第五周学习分类算法支持向量机（SVM），作业要求使用课件描述的数据集完成数据分类。本周选用课件案例②的 **Statlog (German Credit Data)** 数据集（OpenML `data_id=31`，1000 条贷款记录 × 20 个特征，好坏比 7:3）。

### 理论要点

- 几何间隔 \(\gamma_i = y_i(w^Tx_i+b)/\|w\|\)，原始问题 \(\min \frac{1}{2}\|w\|^2\)；
- 软间隔引入松弛变量，参数 \(C\) 权衡间隔宽度与违规数量；
- 核技巧将内积替换为 \(K(x_a,x_b)\)，RBF 核可学习非线性边界；
- 类别不平衡时用 `class_weight='balanced'` 给少数类更大的惩罚。

### 实验设计

文件：[homework-week5/code/credit_svm.py](./homework-week5/code/credit_svm.py)

- 数值特征标准化、类别特征独热编码，全部封装在 `Pipeline` 内防止数据泄漏；
- 对比 RBF 核 SVM（class_weight balanced）、线性 SVM 与决策树，5 折分层交叉验证评估 AUC；
- 指标：CV AUC、测试 AUC、准确率、违约类召回率/精确率、AUC-PR。

本次运行的关键结果约为：

| 指标 | RBF SVM | 线性 SVM | 决策树 |
| --- | ---: | ---: | ---: |
| 测试 AUC | **0.812** | 0.807 | 0.599 |
| 测试 AUC-PR | **0.685** | 0.633 | 0.376 |
| 违约召回率 | 0.567 | 0.800 | 0.667 |

结论：小样本 + 独热后高维稀疏的场景下，RBF 核 SVM 排序能力最强，决策树接近随机；不平衡数据应看 AUC 与少数类召回，而不是准确率。

详细分析见 [homework-week5/analysis.md](./homework-week5/analysis.md)。

## 技术栈

- Python 3
- pandas
- statsmodels
- OLS Multiple Linear Regression
- Variance Inflation Factor (VIF)
- Ridge / Lasso / Elastic Net
- scikit-learn Pipeline 与 10 折交叉验证
- Netflix 数据 Elastic Net 分析
- SVM 分类（线性核 / RBF 核、GridSearchCV、class_weight、ROC / AUC）

安装全部依赖：

```bash
pip install -r requirements.txt
```

---

> 本仓库将按照课程进度持续更新，每周作业统一放入对应的 `homework-weekN` 目录中。
