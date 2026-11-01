# =============================================================
# 信用评分卡项目 · 第 4 步：WOE 编码 → 变量筛选 → 逻辑回归 → 换算成评分卡
# 运行：py -3.13 step4_model.py > step4_output.txt
# =============================================================

# %% 0. 导入库、读数据和第 3 步保存的切分点
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
with open("data/bins.json", encoding="utf-8") as f:
    BINS = json.load(f)

TARGET = "bad"
FEATURES = list(BINS.keys())
CORR_LIMIT = 0.7     # 两个变量的相关系数超过这个值，就只保留 IV 更高的那个


# 下面两个函数和第 3 步完全一样
def assign_bin(x, cuts):
    edges = [-np.inf] + cuts + [np.inf]
    b = pd.cut(x, bins=edges, right=True)
    b = b.cat.add_categories("缺失").fillna("缺失")
    return b


def woe_table(bins, y):
    t = (pd.DataFrame({"bin": bins, "bad": y})
         .groupby("bin", observed=True)["bad"]
         .agg(n="count", n_bad="sum"))
    t["n_good"] = t["n"] - t["n_bad"]
    t["bad_rate"] = t["n_bad"] / t["n"]
    dist_bad = (t["n_bad"] + 0.5) / (t["n_bad"].sum() + 0.5 * len(t))
    dist_good = (t["n_good"] + 0.5) / (t["n_good"].sum() + 0.5 * len(t))
    t["woe"] = np.log(dist_bad / dist_good)
    t["iv_part"] = (dist_bad - dist_good) * t["woe"]
    return t


# =============================================================
# 1. WOE 编码：只用训练集算每一箱的 WOE，再套用到训练集和测试集
# =============================================================

# %% 1.1 在训练集上算 WOE
woe_maps, iv = {}, {}
for col in FEATURES:
    t = woe_table(assign_bin(train[col], BINS[col]), train[TARGET])
    m = t["woe"].copy()
    m.index = m.index.astype(str)
    woe_maps[col] = m
    iv[col] = t["iv_part"].sum()


# %% 1.2 把原始值换成所在箱的 WOE
def to_woe(df):
    out = pd.DataFrame(index=df.index)
    for col in FEATURES:
        b = assign_bin(df[col], BINS[col]).astype(str)
        # 测试集里如果出现训练集没有的箱（比如训练集该变量没有缺失），WOE 记为 0，即"平均风险"
        out[col] = b.map(woe_maps[col]).astype(float).fillna(0.0)
    return out


X_train, X_test = to_woe(train), to_woe(test)
y_train, y_test = train[TARGET], test[TARGET]
print("【1 WOE 编码后的前 5 行（训练集）】")
print(X_train.head().round(3), "\n")


# =============================================================
# 2. 相关性检查：高度相关的两个变量只留一个
# =============================================================

# %% 2.1 相关系数矩阵
corr = X_train.corr()
print("【2.1 WOE 变量之间的相关系数】")
print(corr.round(2), "\n")

# %% 2.2 按 IV 从高到低，遇到和已保留变量高度相关的就剔除
ranked = sorted(FEATURES, key=lambda c: -iv[c])
selected = []
print(f"【2.2 相关性筛选（阈值 {CORR_LIMIT}）】")
for c in ranked:
    clash = [s for s in selected if abs(corr.loc[c, s]) > CORR_LIMIT]
    if clash:
        print(f"  剔除 {c}（IV={iv[c]:.3f}）：和 {clash[0]} 的相关系数 "
              f"{corr.loc[c, clash[0]]:.2f}，保留 IV 更高的 {clash[0]}")
    else:
        selected.append(c)
print(f"  保留：{selected}\n")


# =============================================================
# 3. 逻辑回归 + 系数方向检查
#    WOE 越大风险越高，所以系数都应为正；出现负系数说明和别的变量信息重叠，要剔除重拟
# =============================================================

# %% 3.1 拟合，有负系数就剔除其中 IV 最低的变量，直到全部为正
def fit(cols):
    # C 取很大的值，相当于不加正则化，和传统评分卡的做法一致
    model = LogisticRegression(C=1e6, max_iter=2000)
    model.fit(X_train[cols], y_train)
    return model


print("【3.1 系数方向检查】")
while True:
    model = fit(selected)
    coef = pd.Series(model.coef_[0], index=selected)
    negative = coef[coef <= 0]
    if negative.empty:
        print("  所有系数都为正 ✓\n")
        break
    worst = min(negative.index, key=lambda c: iv[c])
    print(f"  {worst} 的系数为 {coef[worst]:.3f}（负数）→ 剔除后重新拟合")
    selected.remove(worst)

# %% 3.2 最终模型
print("【3.2 最终模型】")
coef_table = pd.DataFrame({"系数": coef, "IV": [iv[c] for c in selected]}).round(4)
print(coef_table.sort_values("IV", ascending=False))
print(f"截距：{model.intercept_[0]:.4f}\n")

# %% 3.3 先粗看一下区分能力（完整评估在第 5 步）
auc_train = roc_auc_score(y_train, model.predict_proba(X_train[selected])[:, 1])
auc_test = roc_auc_score(y_test, model.predict_proba(X_test[selected])[:, 1])
print(f"【3.3 AUC】训练集 {auc_train:.4f}   测试集 {auc_test:.4f}\n")


# =============================================================
# 4. 换算成评分卡
#    分数 = A − B × ln(坏好比)
#    设定：坏好比为 1:50 时得 600 分；坏好比每减半（即好坏比翻倍），分数加 PDO = 20 分
# =============================================================

# %% 4.1 计算 A、B
BASE_SCORE, BASE_ODDS, PDO = 600, 1 / 50, 20
B = PDO / np.log(2)
A = BASE_SCORE + B * np.log(BASE_ODDS)
base_points = A - B * model.intercept_[0]
print(f"【4.1 刻度参数】B = {B:.2f}，A = {A:.2f}，基础分 = {base_points:.0f}\n")

# %% 4.2 每个变量每一箱的分数 = −B × 系数 × WOE（风险高的箱 WOE 为正，所以得分为负，拉低总分）
rows = []
for col in selected:
    for bin_label, w in woe_maps[col].items():
        rows.append({"变量": col, "分箱": bin_label, "WOE": round(w, 4),
                     "分数": int(round(-B * coef[col] * w))})
card = pd.DataFrame(rows)
print("【4.2 评分卡】")
print(f"基础分：{base_points:.0f}")
print(card.to_string(index=False), "\n")

# %% 4.3 用评分卡给每个人打分
points_maps = {col: card[card["变量"] == col].set_index("分箱")["分数"] for col in selected}


def score(df):
    s = pd.Series(round(base_points), index=df.index, dtype=float)
    for col in selected:
        b = assign_bin(df[col], BINS[col]).astype(str)
        s += b.map(points_maps[col]).astype(float).fillna(0)
    return s


train_score, test_score = score(train), score(test)
print("【4.3 训练集：好客户和坏客户的平均分】")
print(pd.DataFrame({"score": train_score, "bad": y_train})
      .groupby("bad")["score"].describe()[["count", "mean", "25%", "50%", "75%"]].round(1), "\n")

# %% 4.4 保存：评分卡（Excel 能直接打开的编码）和每个人的分数，第 5 步评估用
card_out = pd.concat([pd.DataFrame([{"变量": "基础分", "分箱": "", "WOE": None,
                                     "分数": int(round(base_points))}]), card])
card_out.to_csv("data/scorecard.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"bad": y_train, "score": train_score,
              "prob": model.predict_proba(X_train[selected])[:, 1]}).to_csv("data/train_scored.csv", index=False)
pd.DataFrame({"bad": y_test, "score": test_score,
              "prob": model.predict_proba(X_test[selected])[:, 1]}).to_csv("data/test_scored.csv", index=False)
print("已保存 data/scorecard.csv、data/train_scored.csv、data/test_scored.csv")
