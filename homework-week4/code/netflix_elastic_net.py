"""Week 4 - Netflix 数据的 Elastic Net 分析。

数据：Kaggle Netflix Movies and TV Shows（netflix_titles.csv）。
目标：使用影片/剧集的元数据预测 release_year，分析 Elastic Net 的变量收缩与筛选效果。

依赖：pip install numpy pandas scikit-learn matplotlib
"""

from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = "https://raw.githubusercontent.com/MadoDoctor/EDA-Project-Netflix/main/netflix_titles.csv"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, OUTPUT_DIR = ROOT / "data", ROOT / "results"


def load_netflix():
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "netflix_titles.csv"
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def prepare_data(df):
    """把 Netflix 文本元数据转成可用于回归的结构化特征。"""
    out = pd.DataFrame(index=df.index)
    out["type"] = df["type"].fillna("Unknown")
    out["rating"] = df["rating"].fillna("Unknown")
    out["listed_in"] = df["listed_in"].fillna("Unknown").str.split(", ").str[0]
    out["country_count"] = df["country"].fillna("").str.count(",") + (df["country"].fillna("") != "").astype(int)
    out["genre_count"] = df["listed_in"].fillna("").str.count(",") + (df["listed_in"].fillna("") != "").astype(int)
    out["description_length"] = df["description"].fillna("").str.len()
    out["title_length"] = df["title"].fillna("").str.len()
    out["has_director"] = df["director"].notna().astype(int)
    out["has_cast"] = df["cast"].notna().astype(int)
    out["target"] = pd.to_numeric(df["release_year"], errors="coerce")
    return out.dropna(subset=["target"])


def main():
    df = prepare_data(load_netflix())
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    model = ElasticNetCV(alphas=np.logspace(-3, 3, 100), l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], cv=KFold(10, shuffle=True, random_state=42), max_iter=100000, random_state=42)
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    coef = pipe.named_steps["model"].coef_.ravel()
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    selected = pd.DataFrame({"feature": names, "coefficient": coef})
    selected["absolute_coefficient"] = selected["coefficient"].abs()
    selected = selected[selected["absolute_coefficient"] > 1e-8].sort_values("absolute_coefficient", ascending=False)
    OUTPUT_DIR.mkdir(exist_ok=True)
    selected.to_csv(OUTPUT_DIR / "netflix_elastic_net_selected_features.csv", index=False)
    metrics = {"samples": len(df), "alpha": model.alpha_, "l1_ratio": model.l1_ratio_, "test_RMSE": mean_squared_error(y_test, pred) ** 0.5, "test_MAE": mean_absolute_error(y_test, pred), "test_R2": r2_score(y_test, pred), "nonzero_features": len(selected)}
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "netflix_elastic_net_metrics.csv", index=False)
    print("Netflix 样本量：", len(df))
    print("Elastic Net 最优 alpha：", f"{model.alpha_:.6f}")
    print("Elastic Net 最优 l1_ratio：", f"{model.l1_ratio_:.2f}")
    print(f"测试集 RMSE={metrics['test_RMSE']:.4f}, MAE={metrics['test_MAE']:.4f}, R²={metrics['test_R2']:.4f}")
    print("非零特征数：", len(selected))
    print("\n绝对系数最大的特征：")
    print(selected.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    plt.figure(figsize=(10, 5))
    top = selected.head(15).sort_values("coefficient")
    plt.barh(top["feature"], top["coefficient"])
    plt.title("Netflix Elastic Net：绝对值最大的非零系数")
    plt.xlabel("标准化后的回归系数")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "netflix_elastic_net_coefficients.png", dpi=160)


if __name__ == "__main__":
    main()
