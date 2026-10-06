# Python 数据分析作业 Week 3

## 作业 1：正交设计下 Ridge 与 OLS 的关系

考虑不含截距的线性回归，并假设设计矩阵满足正交条件

\[
X^T X=I.
\]

OLS 估计量为

\[
\widehat{\boldsymbol\beta}_{OLS}
  =(X^TX)^{-1}X^Ty=X^Ty.
\]

Ridge 估计量是最小化

\[
\|y-X\boldsymbol\beta\|_2^2+\lambda\|\boldsymbol\beta\|_2^2
\]

所得的闭合解：

\[
\widehat{\boldsymbol\beta}_{Ridge}
  =(X^TX+\lambda I)^{-1}X^Ty.
\]

代入 \(X^TX=I\)：

\[
\begin{aligned}
\widehat{\boldsymbol\beta}_{Ridge}
  &=(I+\lambda I)^{-1}X^Ty\\
  &=\frac{1}{1+\lambda}X^Ty\\
  &=\frac{1}{1+\lambda}\widehat{\boldsymbol\beta}_{OLS}.
\end{aligned}
\]

因此，在正交设计下，Ridge 对每个 OLS 系数进行相同倍数的收缩。\(\lambda=0\) 时恢复 OLS；\(\lambda>0\) 时系数绝对值变小，但通常不会被精确压成 0。

## 作业 3：为什么正则化前必须 Z-score 标准化？

对每个解释变量进行

\[
z_j=\frac{x_j-\bar{x}_j}{s_j}
\]

变换后，各变量大致具有均值 0、标准差 1。这样做有三个重要原因：

1. **保证惩罚公平。** Ridge 惩罚 \(\lambda\sum_j\beta_j^2\)，Lasso 惩罚 \(\lambda\sum_j|\beta_j|\)。如果两个变量使用不同单位，系数的大小会受到量纲影响，而不是只反映变量的重要程度。
2. **避免量纲大的变量被错误地轻罚。** 例如将收入从“元”改成“千元”，对应系数会缩小约 1000 倍，但模型含义没有改变。未标准化时，惩罚会随着单位变化，导致结果不具备尺度不变性。
3. **改善数值计算。** 不同变量尺度差异很大时，目标函数等高线会非常狭长，优化算法收敛更慢，也更容易受数值精度影响。

未标准化的后果是：大尺度变量的系数天然偏小，可能受到相对更强的收缩；小尺度变量的系数可能较大，受到相对更弱的收缩。于是 Ridge、Lasso 或 Elastic Net 的变量比较和变量筛选会被单位“绑架”。此外，截距通常不应被正则化；在实际建模中应将中心化、标准化和正则化放入同一个 `Pipeline`，只对训练集拟合标准化参数，避免数据泄漏。

## 作业 4：Ridge、Lasso 与 Elastic Net 实战

实战代码见：

```text
homework-week3/code/regularized_regression_comparison.py
```

代码使用公开的 ISLR `Hitters` 数据集，删除 `Salary` 缺失行，以 `Salary` 为响应变量，并对分类变量进行 one-hot 编码。三个模型分别使用 `RidgeCV`、`LassoCV` 和 `ElasticNetCV`，交叉验证折数均为 10。预处理和模型放在同一个 Pipeline 中，确保标准化只使用训练集信息。

程序会：

- 固定随机种子划分训练集和测试集；
- 输出三种模型的测试集 RMSE 和非零变量数；
- 绘制三种模型的系数路径图；
- 根据交叉验证误差的最小值和 1-SE 阈值，报告更简单的候选正则化强度。

1-SE 法则的思想是：先找到最小交叉验证误差 \(MSE_{min}\)，再选择满足

\[
MSE(\lambda)\leq MSE_{min}+SE_{min}
\]

的模型中正则化更强、通常也更简单的那个。这样做通常会牺牲很小的预测性能，换取更低的模型复杂度和更好的稳定性。对 Lasso 和 Elastic Net，1-SE 选择往往会保留更少的非零变量；对 Ridge，系数一般仍然全部非零，因此“更简单”主要体现为更强收缩，而不是变量删除。
