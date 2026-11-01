# =============================================================
# 信用评分卡项目 · 第 2 步：诊断问题数据 → 清洗 → 划分训练集和测试集
# 运行前先装：py -3.13 -m pip install scikit-learn
# 运行：py -3.13 step2_clean.py > step2_output.txt
# =============================================================

# %% 0. 导入库、读数据（和第 1 步一样）
import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

DATA = "data/cs-training.csv"
rename = {
    "SeriousDlqin2yrs": "bad",
    "RevolvingUtilizationOfUnsecuredLines": "revol_util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "dpd30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "income",
    "NumberOfOpenCreditLinesAndLoans": "open_lines",
    "NumberOfTimes90DaysLate": "dpd90",
    "NumberRealEstateLoansOrLines": "re_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "dpd60_89",
    "NumberOfDependents": "dependents",
}
df = pd.read_csv(DATA, index_col=0).rename(columns=rename)

con = sqlite3.connect(":memory:")
df.to_sql("raw", con, index=False)


def q(sql):
    return pd.read_sql(sql, con)


# =============================================================
# A. 先诊断：这些"问题数据"背后是什么样的人？
#    清洗之前先弄清楚问题数据的风险高低，才知道该怎么处理
# =============================================================

# %% A1. 缺失的人，坏客户比例和其他人一样吗？
print("【A1 收入是否缺失 × 坏客户比例】")
print(q("""
    SELECT CASE WHEN income IS NULL THEN '收入缺失' ELSE '收入有值' END AS grp,
           COUNT(*) AS n,
           ROUND(AVG(bad), 4) AS bad_rate
    FROM raw GROUP BY grp
""").to_string(index=False), "\n")

print("【A1 供养人数是否缺失 × 坏客户比例】")
print(q("""
    SELECT CASE WHEN dependents IS NULL THEN '供养人数缺失' ELSE '供养人数有值' END AS grp,
           COUNT(*) AS n,
           ROUND(AVG(bad), 4) AS bad_rate
    FROM raw GROUP BY grp
""").to_string(index=False), "\n")

# %% A2. 逾期次数是 96/98 的人：是不是同一批人？风险如何？
print("【A2 96/98 是否为同一批人】")
print(q("""
    SELECT SUM(CASE WHEN dpd30_59 IN (96, 98) THEN 1 ELSE 0 END)  AS in_dpd30_59,
           SUM(CASE WHEN dpd60_89 IN (96, 98) THEN 1 ELSE 0 END)  AS in_dpd60_89,
           SUM(CASE WHEN dpd90    IN (96, 98) THEN 1 ELSE 0 END)  AS in_dpd90,
           SUM(CASE WHEN dpd30_59 IN (96, 98) AND dpd60_89 IN (96, 98)
                     AND dpd90 IN (96, 98) THEN 1 ELSE 0 END)     AS in_all_three
    FROM raw
""").to_string(index=False), "\n")

print("【A2 96/98 人群 × 坏客户比例】")
print(q("""
    SELECT CASE WHEN dpd30_59 IN (96, 98) THEN '特殊编码 96/98' ELSE '正常取值' END AS grp,
           COUNT(*) AS n,
           ROUND(AVG(bad), 4) AS bad_rate
    FROM raw GROUP BY grp
""").to_string(index=False), "\n")

# %% A3. 额度使用率分段：超过 1 的人风险如何？超过 10 的呢？
print("【A3 额度使用率分段 × 坏客户比例】")
print(q("""
    SELECT CASE WHEN revol_util <= 0.5 THEN '1) 0 - 0.5'
                WHEN revol_util <= 1   THEN '2) 0.5 - 1'
                WHEN revol_util <= 2   THEN '3) 1 - 2'
                WHEN revol_util <= 10  THEN '4) 2 - 10'
                ELSE                        '5) > 10' END AS util_band,
           COUNT(*) AS n,
           ROUND(AVG(bad), 4) AS bad_rate
    FROM raw GROUP BY util_band ORDER BY util_band
""").to_string(index=False), "\n")

# %% A4. 收入缺失时，debt_ratio 还像一个"比率"吗？（SQLite 没有中位数函数，这里用 pandas）
print("【A4 收入是否缺失 × debt_ratio 的分布】")
grp = np.where(df["income"].isna(), "收入缺失", "收入有值")
print(df.groupby(grp)["debt_ratio"].describe(percentiles=[0.25, 0.5, 0.75]).round(2), "\n")


# =============================================================
# B. 清洗
#    这里的规则都是固定的业务判断，不是从数据里"学"出来的参数，
#    所以可以在划分训练集、测试集之前做，不会造成数据泄漏
# =============================================================
clean = df.copy()

# %% B1. 年龄为 0 是录入错误，删掉这条记录
n_before = len(clean)
clean = clean[clean["age"] > 0]
print(f"【B1】删除年龄为 0 的记录：{n_before - len(clean)} 条\n")

# %% B2. 逾期次数里的 96/98 是特殊编码，不是真实次数 → 当作缺失
for col in ["dpd30_59", "dpd60_89", "dpd90"]:
    clean.loc[clean[col].isin([96, 98]), col] = np.nan

# %% B3. 额度使用率大于 10 不可能是真实的使用率 → 当作缺失
#       （阈值先定 10，看完 A3 的结果再确认是否合理）
clean.loc[clean["revol_util"] > 10, "revol_util"] = np.nan

# %% B4. 收入缺失时，负债比没法正确计算（A4 可以验证）→ 也当作缺失
clean.loc[clean["income"].isna(), "debt_ratio"] = np.nan

# %% B5. 清洗后的缺失情况
#       缺失值先保留，不填补：第 3 步分箱时，缺失会单独成为一组
print("【B5 清洗后各列缺失数】")
print(clean.isna().sum(), "\n")


# =============================================================
# C. 划分训练集（70%）和测试集（30%）
#    stratify=bad：分层抽样，保证两边的坏客户比例一致
#    random_state=42：固定随机种子，每次运行结果都一样，方便复现
# =============================================================

# %% C1. 划分并检查
train, test = train_test_split(
    clean, test_size=0.3, stratify=clean["bad"], random_state=42
)
print("【C1 训练集 / 测试集】")
print(pd.DataFrame({
    "n": [len(train), len(test)],
    "bad_rate": [round(train["bad"].mean(), 4), round(test["bad"].mean(), 4)],
}, index=["train", "test"]), "\n")

# %% C2. 保存，第 3 步直接读这两个文件
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("已保存 data/train.csv 和 data/test.csv")
