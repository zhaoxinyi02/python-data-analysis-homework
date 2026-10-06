"""Week 3 - Ridge、Lasso 与 Elastic Net 对比（Hitters 数据集）。

依赖：pip install numpy pandas scikit-learn matplotlib
"""

from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Hitters.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"


def load_data():
    """下载并清理 Hitters 数据集。"""
    df = pd.read_csv(DATA_URL)
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df]).dropna(subset=["Salary"])
    return df


def one_se_alpha(model):
    """从 sklearn CV 的逐折结果中返回 1-SE 规则选择的 alpha。"""
    path = np.asarray(model.mse_path_)
    # LassoCV: (n_alphas, n_folds)；ElasticNetCV: (n_alphas, n_folds, n_l1_ratio)。
    # Elastic Net 先在 l1_ratio 维度上取最优（最小）CV MSE。
    if path.ndim == 3:
        path = path.min(axis=2)
    mean_score = path.mean(axis=1)
    se = path.std(axis=1, ddof=1) / np.sqrt(path.shape[1])
    best = int(np.argmin(mean_score))
    threshold = mean_score[best] + se[best]
    eligible = np.flatnonzero(mean_score <= threshold)
    # alpha 网格通常从大到小；在允许范围内选择更大的 alpha（更强正则化）。
    chosen = int(eligible[np.argmax(model.alphas_[eligible])])
    return float(model.alphas_[chosen]), float(mean_score[best]), float(se[best])


def build_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore")), ("scale", StandardScaler(with_mean=False))]), categorical),
        ]
    )


def main():
    df = load_data()
    X, y = df.drop(columns="Salary"), df["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 4, 120)
    models = {
        "Ridge": RidgeCV(alphas=alphas, cv=cv),
        "Lasso": LassoCV(alphas=alphas, cv=cv, max_iter=100000, random_state=42),
        "Elastic Net": ElasticNetCV(alphas=alphas, l1_ratio=[0.2, 0.5, 0.8], cv=cv, max_iter=100000, random_state=42),
    }
    preprocessor = build_preprocessor(X_train)
    results, paths = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        coef = pipe.named_steps["model"].coef_.ravel()
        results.append({"Model": name, "Test RMSE": mean_squared_error(y_test, pred) ** 0.5, "Non-zero variables": int(np.sum(np.abs(coef) > 1e-8)), "Alpha": float(pipe.named_steps["model"].alpha_)})
        paths[name] = (pipe, coef)

    OUTPUT_DIR.mkdir(exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "regularized_regression_metrics.csv", index=False)
    print("Hitters 样本量：", len(df))
    print("\n测试集结果：")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n1-SE 法则参考：")
    for name in ["Lasso", "Elastic Net"]:
        estimator = paths[name][0].named_steps["model"]
        alpha_1se, mse_min, se = one_se_alpha(estimator)
        print(f"{name}: CV MSE 最小值={mse_min:.2f}, SE={se:.2f}, 1-SE 推荐 alpha={alpha_1se:.5f}")
    print("Ridge 的系数通常全部非零，因此 1-SE 体现为更强收缩而非删除变量。")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    feature_count = len(paths["Ridge"][1])
    for ax, (name, (pipe, _)) in zip(axes, paths.items()):
        estimator = pipe.named_steps["model"]
        grid = estimator.alphas_ if hasattr(estimator, "alphas_") else alphas
        # 用固定标准化后的训练矩阵展示从强到弱正则化时的系数路径。
        Xtr = pipe.named_steps["preprocess"].transform(X_train)
        if name == "Ridge":
            values = np.array([RidgeCV(alphas=[a]).fit(Xtr, y_train).coef_ for a in grid])
        elif name == "Lasso":
            values = np.array([LassoCV(alphas=[a], cv=3, max_iter=100000).fit(Xtr, y_train).coef_ for a in grid])
        else:
            values = np.array([ElasticNetCV(alphas=[a], l1_ratio=[estimator.l1_ratio_], cv=3, max_iter=100000).fit(Xtr, y_train).coef_ for a in grid])
        ax.plot(np.log10(grid), values, linewidth=0.7, alpha=0.65)
        ax.axvline(np.log10(estimator.alpha_), color="black", linestyle="--", linewidth=1, label="CV alpha")
        ax.set_title(name)
        ax.set_xlabel("log10(alpha)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Coefficient")
    fig.suptitle("Hitters：正则化模型系数路径")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regularized_coefficient_paths.png", dpi=160)
    print(f"\n结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
