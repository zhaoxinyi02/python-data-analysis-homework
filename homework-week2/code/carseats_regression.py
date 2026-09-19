"""
Week 2 - Carseats 多元线性回归

任务：
1. 以 Sales 为响应变量；
2. 使用 Price、Income、Advertising 和分类变量 ShelveLoc 建立多元线性回归；
3. 输出模型拟合报告；
4. 指出 ShelveLoc 的基准组；
5. 解释 ShelveLoc[Good] 系数；
6. 计算各解释变量（含 ShelveLoc 虚拟变量）的 VIF，并判断是否有明显多重共线性。

依赖：
    pip install pandas statsmodels
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_URLS = [
    "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv",
    "https://raw.githubusercontent.com/selva86/datasets/master/Carseats.csv",
]


def load_carseats():
    """从公开数据源加载 Carseats 数据集。"""
    last_error = None

    for url in DATA_URLS:
        try:
            df = pd.read_csv(url)
            required = {"Sales", "Price", "Income", "Advertising", "ShelveLoc"}
            if required.issubset(df.columns):
                print(f"数据加载成功：{url}")
                print(f"样本量：{len(df)}")
                return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Carseats 数据集加载失败，请检查网络连接。"
    ) from last_error


def main():
    df = load_carseats()

    # 明确指定 ShelveLoc 的三个水平，并将 Bad 设为基准组。
    df["ShelveLoc"] = pd.Categorical(
        df["ShelveLoc"],
        categories=["Bad", "Medium", "Good"]
    )

    formula = (
        "Sales ~ Price + Income + Advertising + "
        "C(ShelveLoc, Treatment(reference='Bad'))"
    )

    model = smf.ols(formula=formula, data=df).fit()

    print("\n" + "=" * 80)
    print("OLS 模型拟合报告")
    print("=" * 80)
    print(model.summary())

    print("\n" + "=" * 80)
    print("ShelveLoc 基准组与 Good 系数解释")
    print("=" * 80)

    print("ShelveLoc 的基准组：Bad")

    good_name = "C(ShelveLoc, Treatment(reference='Bad'))[T.Good]"
    good_coef = model.params[good_name]

    print(f"ShelveLoc[Good] 的估计系数：{good_coef:.4f}")
    print(
        "解释：在 Price、Income、Advertising 保持相同的条件下，"
        f"ShelveLoc=Good 的商店相对于基准组 ShelveLoc=Bad 的商店，"
        f"预测 Sales 平均高 {good_coef:.4f} 千件。"
    )
    print(
        f"也就是模型预测约多销售 {good_coef * 1000:.0f} 件儿童座椅。"
        "这里是回归模型中的条件相关关系，不应直接解释为因果效应。"
    )

    print("\n" + "=" * 80)
    print("VIF 多重共线性检验")
    print("=" * 80)

    # 使用与回归模型完全相同的设计矩阵。
    # 分类变量 ShelveLoc 被展开为两个虚拟变量，Bad 为基准组。
    exog = pd.DataFrame(
        model.model.exog,
        columns=model.model.exog_names
    )

    vif_rows = []
    for i, name in enumerate(exog.columns):
        if name == "Intercept":
            continue
        vif_rows.append(
            {
                "Variable": name,
                "VIF": variance_inflation_factor(exog.values, i),
            }
        )

    vif_df = pd.DataFrame(vif_rows)
    print(vif_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    max_vif = vif_df["VIF"].max()
    print(f"\n最大 VIF = {max_vif:.4f}")

    if max_vif < 5:
        print("结论：各变量 VIF 均低于 5，没有明显的多重共线性风险。")
    elif max_vif < 10:
        print("结论：存在一定程度的多重共线性，需要进一步关注。")
    else:
        print("结论：存在较强的多重共线性风险。")

    print("\n参考结果（本数据集应接近）：")
    print("R^2 ≈ 0.627")
    print("ShelveLoc[Good] 系数 ≈ 4.836")
    print(
        "VIF：Price≈1.008，Income≈1.012，Advertising≈1.009，"
        "ShelveLoc 两个虚拟变量≈1.49～1.50。"
    )


if __name__ == "__main__":
    main()
