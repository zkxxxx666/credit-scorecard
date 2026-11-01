# =============================================================
# 信用评分卡项目 · 第 5 步：评估模型 + 分数线模拟 + 和梯度提升树（GBDT）对比
# 运行前先装：py -3.13 -m pip install matplotlib
# 运行：py -3.13 step5_evaluate.py > step5_output.txt
# 图片保存在 figures/ 文件夹，之后放进 GitHub 的说明文档
# =============================================================

# %% 0. 导入库、读第 4 步打好分的数据
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # 只保存图片，不弹窗口
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("figures", exist_ok=True)

train_s = pd.read_csv("data/train_scored.csv")
test_s = pd.read_csv("data/test_scored.csv")


# =============================================================
# 1. AUC 和 KS
#    分数越高风险越低，所以算指标时用 "-分数" 当作风险
# =============================================================

# %% 1.1 定义 KS
def ks_stat(y, score):
    """KS = 在所有可能的分数线里，(坏客户累计占比 − 好客户累计占比) 的最大值"""
    fpr, tpr, _ = roc_curve(y, -score)
    return float(np.max(tpr - fpr))


# %% 1.2 训练集和测试集的 AUC、KS
print("【1 评分卡的 AUC 和 KS】")
metrics = pd.DataFrame({
    "AUC": [roc_auc_score(d["bad"], -d["score"]) for d in (train_s, test_s)],
    "KS": [ks_stat(d["bad"], d["score"]) for d in (train_s, test_s)],
}, index=["训练集", "测试集"]).round(4)
print(metrics, "\n")


# =============================================================
# 2. 十分位表：按测试集分数从低到高等分成 10 组
#    看每组的坏客户比例，以及从低分往上累计能抓到多少坏客户
# =============================================================

# %% 2.1 十分位表
t = test_s.copy()
t["组"] = pd.qcut(t["score"].rank(method="first"), 10, labels=range(1, 11))
dec = t.groupby("组", observed=True).agg(
    最低分=("score", "min"), 最高分=("score", "max"),
    人数=("bad", "size"), 坏客户数=("bad", "sum"))
dec["坏客户比例"] = dec["坏客户数"] / dec["人数"]
dec["累计坏客户占比"] = dec["坏客户数"].cumsum() / dec["坏客户数"].sum()
good = dec["人数"] - dec["坏客户数"]
dec["累计好客户占比"] = good.cumsum() / good.sum()
dec["KS"] = dec["累计坏客户占比"] - dec["累计好客户占比"]
print("【2 十分位表（测试集，第 1 组分数最低）】")
print(dec.round(4).to_string(), "\n")


# =============================================================
# 3. PSI：训练集和测试集的分数分布是否一致
#    这里是随机划分，所以 PSI 应该接近 0；实际工作中，PSI 用来比较
#    "建模时的客群"和"上线后每个月的新客群"，监控模型是否还适用
# =============================================================

# %% 3.1 计算 PSI
def psi(expected, actual, n_bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges).value_counts(normalize=True, sort=False).values
    a = pd.cut(actual, edges).value_counts(normalize=True, sort=False).values
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


print(f"【3 PSI（训练集 vs 测试集）】{psi(train_s['score'], test_s['score']):.4f}\n")


# =============================================================
# 4. 分数线模拟：低于分数线的拒绝，看通过率和通过人群的坏客户比例
# =============================================================

# %% 4.1 不同分数线下的通过率和坏账率
rows = []
for cut in range(500, 621, 10):
    approved = test_s[test_s["score"] >= cut]
    rejected = test_s[test_s["score"] < cut]
    rows.append({
        "分数线": cut,
        "通过率": len(approved) / len(test_s),
        "通过人群坏客户比例": approved["bad"].mean(),
        "拦下的坏客户占全部坏客户": rejected["bad"].sum() / test_s["bad"].sum(),
    })
print("【4 分数线模拟（测试集）】")
print(f"不设分数线时，全部通过，坏客户比例为 {test_s['bad'].mean():.4f}")
print(pd.DataFrame(rows).round(4).to_string(index=False), "\n")


# =============================================================
# 5. 和梯度提升树（GBDT）对比
#    直接用清洗后的原始变量，不分箱、不做 WOE；它能自己处理缺失值
# =============================================================

# %% 5.1 训练 GBDT
train_raw = pd.read_csv("data/train.csv")
test_raw = pd.read_csv("data/test.csv")
X_cols = [c for c in train_raw.columns if c != "bad"]
gbdt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                      early_stopping=True, random_state=42)
gbdt.fit(train_raw[X_cols], train_raw["bad"])
p_gbdt = gbdt.predict_proba(test_raw[X_cols])[:, 1]

# %% 5.2 对比（都在测试集上）
print("【5 评分卡 vs GBDT（测试集）】")
compare = pd.DataFrame({
    "AUC": [roc_auc_score(test_s["bad"], -test_s["score"]),
            roc_auc_score(test_raw["bad"], p_gbdt)],
    "KS": [ks_stat(test_s["bad"], test_s["score"]),
           ks_stat(test_raw["bad"], -p_gbdt)],
    "可解释性": ["每个变量每一段的得分都能看到", "黑箱，需要额外工具解释"],
}, index=["评分卡（逻辑回归）", "GBDT"])
print(compare.round(4).to_string(), "\n")


# =============================================================
# 6. 画图，保存到 figures/
# =============================================================

# %% 6.1 ROC 曲线
fig, ax = plt.subplots(figsize=(6, 5))
for name, y, risk in [("评分卡", test_s["bad"], -test_s["score"]),
                      ("GBDT", test_raw["bad"], p_gbdt)]:
    fpr, tpr, _ = roc_curve(y, risk)
    ax.plot(fpr, tpr, label=f"{name}（AUC = {roc_auc_score(y, risk):.3f}）")
ax.plot([0, 1], [0, 1], "--", color="grey", label="随机猜测")
ax.set_xlabel("好客户被误拦的比例")
ax.set_ylabel("坏客户被拦下的比例")
ax.set_title("ROC 曲线（测试集）")
ax.legend()
fig.tight_layout()
fig.savefig("figures/roc.png", dpi=150)

# %% 6.2 KS 曲线：横轴是分数线，两条线分别是低于该分数的坏客户、好客户累计占比
s = np.sort(test_s["score"].unique())
bad_cdf = [(test_s.loc[test_s["bad"] == 1, "score"] <= v).mean() for v in s]
good_cdf = [(test_s.loc[test_s["bad"] == 0, "score"] <= v).mean() for v in s]
gap = np.array(bad_cdf) - np.array(good_cdf)
i = int(np.argmax(gap))
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(s, bad_cdf, label="坏客户累计占比")
ax.plot(s, good_cdf, label="好客户累计占比")
ax.vlines(s[i], good_cdf[i], bad_cdf[i], colors="red", linestyles="--",
          label=f"KS = {gap[i]:.3f}（分数 {s[i]:.0f}）")
ax.set_xlabel("分数")
ax.set_ylabel("累计占比")
ax.set_title("KS 曲线（测试集）")
ax.legend()
fig.tight_layout()
fig.savefig("figures/ks.png", dpi=150)

# %% 6.3 好、坏客户的分数分布
fig, ax = plt.subplots(figsize=(6, 5))
bins = np.linspace(test_s["score"].min(), test_s["score"].max(), 40)
ax.hist(test_s.loc[test_s["bad"] == 0, "score"], bins=bins, alpha=0.6, density=True, label="好客户")
ax.hist(test_s.loc[test_s["bad"] == 1, "score"], bins=bins, alpha=0.6, density=True, label="坏客户")
ax.set_xlabel("分数")
ax.set_ylabel("密度")
ax.set_title("好坏客户分数分布（测试集）")
ax.legend()
fig.tight_layout()
fig.savefig("figures/score_dist.png", dpi=150)

print("图片已保存：figures/roc.png、figures/ks.png、figures/score_dist.png")
