# =============================================================
# 信用评分卡项目 · 第 3 步：分箱 → 计算 WOE 和 IV → 筛选变量
# 只用训练集来找切分点、算 WOE，测试集留到最后检验，避免数据泄漏
# 运行：py -3.13 step3_woe_iv.py > step3_output.txt
# =============================================================

# %% 0. 导入库、读训练集
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

train = pd.read_csv("data/train.csv")
TARGET = "bad"
FEATURES = ["revol_util", "age", "dpd30_59", "debt_ratio", "income",
            "open_lines", "dpd90", "re_loans", "dpd60_89", "dependents"]

MAX_BINS = 6        # 每个变量最多分几段
MIN_FRAC = 0.05     # 每段至少占非缺失样本的 5%，避免出现样本太少、比例不可靠的段

# 人工调整的切分点：自动分箱后人工检查，对不合理的变量手动指定
# 没写在这里的变量，继续用决策树自动找的切分点
MANUAL_CUTS = {
    "dpd60_89":   [0.5],                              # 0 次 / 1 次以上（逾期过的人只占约 4.9%，被 5% 的规则卡住了）
    "income":     [3332.5, 5563.5, 6666.5, 10284.0],  # 合并最低的两箱，消除拐弯
    "debt_ratio": [0.35, 0.5, 0.7],                   # 合并两头，让风险随负债比单调上升
    "re_loans":   [0.5, 2.5],                         # 0 套 / 1–2 套 / 3 套以上，保留有业务逻辑的 U 型
}


# %% 1. 用决策树找切分点
def tree_cuts(x, y):
    """在一个变量上训练一棵浅决策树，用它的分裂点作为分箱的切分点。缺失值不参与。"""
    mask = x.notna()
    tree = DecisionTreeClassifier(max_leaf_nodes=MAX_BINS,
                                  min_samples_leaf=MIN_FRAC,
                                  random_state=42)
    tree.fit(x[mask].to_frame(), y[mask])
    # 叶子节点的 feature 是负数，内部节点（真正发生分裂的地方）才是 >= 0
    t = tree.tree_
    return sorted(float(v) for v in t.threshold[t.feature >= 0])


# %% 2. 按切分点分箱，缺失值单独一箱
def assign_bin(x, cuts):
    edges = [-np.inf] + cuts + [np.inf]
    b = pd.cut(x, bins=edges, right=True)          # 区间是左开右闭 (a, b]
    b = b.cat.add_categories("缺失").fillna("缺失")
    return b


# %% 3. 计算每一箱的 WOE 和整体 IV
def woe_table(bins, y):
    t = (pd.DataFrame({"bin": bins, "bad": y})
         .groupby("bin", observed=True)["bad"]
         .agg(n="count", n_bad="sum"))
    t["n_good"] = t["n"] - t["n_bad"]
    t["bad_rate"] = t["n_bad"] / t["n"]
    # 各加 0.5 做平滑，避免某一箱没有坏客户时出现 log(0)
    dist_bad = (t["n_bad"] + 0.5) / (t["n_bad"].sum() + 0.5 * len(t))
    dist_good = (t["n_good"] + 0.5) / (t["n_good"].sum() + 0.5 * len(t))
    t["woe"] = np.log(dist_bad / dist_good)        # 大于 0：风险高于平均
    t["iv_part"] = (dist_bad - dist_good) * t["woe"]
    return t


def iv_strength(iv):
    if iv < 0.02:
        return "没用"
    if iv < 0.1:
        return "弱"
    if iv < 0.3:
        return "中等"
    if iv < 0.5:
        return "强"
    return "很强"


def is_monotonic(t):
    """不含缺失箱时，坏客户比例是否随变量单调变化"""
    r = t.loc[t.index != "缺失", "bad_rate"].values
    if len(r) <= 2:
        return "是"
    d = np.diff(r)
    return "是" if (d >= 0).all() or (d <= 0).all() else "否"


# %% 4. 对每个变量分箱，打印 WOE 表
bins_spec, summary = {}, []
for col in FEATURES:
    manual = col in MANUAL_CUTS
    cuts = MANUAL_CUTS[col] if manual else tree_cuts(train[col], train[TARGET])
    t = woe_table(assign_bin(train[col], cuts), train[TARGET])
    iv = t["iv_part"].sum()
    bins_spec[col] = cuts
    summary.append({"变量": col, "IV": round(iv, 4), "预测能力": iv_strength(iv),
                    "箱数": len(t), "坏客户比例单调": is_monotonic(t),
                    "分箱方式": "手动" if manual else "自动"})

    print(f"==================== {col}   IV = {iv:.4f}（{iv_strength(iv)}）"
          f"{'  【手动分箱】' if manual else ''}")
    show = t[["n", "n_bad", "bad_rate", "woe", "iv_part"]].copy()
    show.index = show.index.astype(str)
    print(show.round(4).to_string(), "\n")

# %% 5. IV 排名
print("==================== 【IV 排名】")
iv_rank = pd.DataFrame(summary).sort_values("IV", ascending=False)
print(iv_rank.to_string(index=False), "\n")

# %% 6. 保存切分点，第 4 步建模时直接用
with open("data/bins.json", "w", encoding="utf-8") as f:
    json.dump(bins_spec, f, ensure_ascii=False, indent=2)
print("切分点已保存到 data/bins.json")
