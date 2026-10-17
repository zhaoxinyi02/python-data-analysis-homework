"""Week 5：German Credit 信用数据的 SVM 分类（课件案例②）。

数据来自 OpenML 上的 Statlog (German Credit Data) 数据集（data_id=31），
共 1000 条贷款记录、20 个特征（7 个数值 + 13 个类别），目标变量 good/bad，
其中 bad（违约）占 30%，属于类别不平衡的二分类问题。

实验思路：
1. 类别特征做独热编码，数值特征做 Z-score 标准化（SVM 对尺度敏感）；
2. 对比三个模型：线性 SVM、RBF 核 SVM（class_weight='balanced'）、决策树；
3. 用 5 折分层交叉验证评估 AUC，再在独立测试集上报告 AUC、准确率与
   违约类（少数类）的召回率 / 精确率。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.metrics import (accuracy_score, average_precision_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
RANDOM_STATE = 42


def load_data():
    data = fetch_openml(data_id=31, as_frame=True)  # German Credit
    X = data.data
    y = (data.target == "bad").astype(int)  # 1 = 违约（正类、少数类）
    return X, y


def build_pipeline(model):
    # 数值列标准化、类别列独热编码；全部放在 Pipeline 内，交叉验证时每折独立拟合
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


if __name__ == "__main__":
    X, y = load_data()
    print(f"样本量 {len(X)}，特征 {X.shape[1]} 个，违约占比 {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {
        # 课件案例②的推荐方案：RBF 核 + 类别权重调整，给少数类（违约）更大的惩罚
        "RBF SVM (class_weight=balanced)":
            SVC(kernel="rbf", C=5, gamma=0.1, class_weight="balanced", random_state=RANDOM_STATE),
        "linear SVM (C=0.1)":
            LinearSVC(C=0.1, class_weight="balanced", max_iter=50000),
        "decision tree (depth=5)":
            DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, curves = [], {}
    print("\n5 折分层交叉验证（AUC）：")
    for name, model in models.items():
        pipe = build_pipeline(model)
        auc_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        if hasattr(model, "decision_function"):
            score = pipe.decision_function(X_test)
        else:
            score = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "CV_AUC_mean": auc_cv.mean(), "CV_AUC_std": auc_cv.std(),
            "test_AUC": roc_auc_score(y_test, score),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_recall_bad": recall_score(y_test, pred),
            "test_precision_bad": precision_score(y_test, pred, zero_division=0),
            "test_AUC_PR": average_precision_score(y_test, score),
        })
        curves[name] = roc_curve(y_test, score)
        print(f"  {name:36s} CV AUC = {auc_cv.mean():.3f} ± {auc_cv.std():.3f}")

    OUT.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "credit_svm_metrics.csv", index=False)

    plt.figure(figsize=(7, 5))
    for name, (fpr, tpr, _) in curves.items():
        auc = frame.loc[frame["model"] == name, "test_AUC"].iloc[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False positive rate (bad loans)"); plt.ylabel("True positive rate")
    plt.title("German Credit: ROC curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "credit_svm_roc.png", dpi=160)

    print("\n测试集结果：")
    print(frame.round(4).to_string(index=False))
    print("\n结论提示：class_weight='balanced' 让少数类获得更大的 C，等价于给违约类"
          "更大的错分惩罚；评估不平衡数据应看 AUC 与少数类召回，而不是只看准确率。")
