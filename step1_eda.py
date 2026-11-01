# =============================================================
# 信用评分卡项目 · 第 1 步：读入数据，初步认识数据
# 数据：Kaggle "Give Me Some Credit" 的 cs-training.csv
# 运行方式：VS Code 或 Jupyter 里按 "# %%" 逐块运行；也可以直接 python step1_eda.py
# 需要的库：pandas（sqlite3 是 Python 自带的，不用装）
# =============================================================

# %% 0. 导入库
import sqlite3
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

DATA = "data/cs-training.csv"   # 把下载的文件放到 data/ 文件夹里


def q(sql):
    """执行一段 SQL，返回 DataFrame"""
    return pd.read_sql(sql, con)


# %% 1. 读入数据，并把列名改成短一点、好写 SQL 的名字
# 第一列是没有列名的行号，用 index_col=0 去掉
df = pd.read_csv(DATA, index_col=0)

rename = {
    "SeriousDlqin2yrs": "bad",                            # 目标：未来两年内是否出现 90 天以上严重逾期（1=坏客户）
    "RevolvingUtilizationOfUnsecuredLines": "revol_util", # 信用卡等无抵押循环额度的使用率（已用额度 / 总额度）
    "age": "age",                                         # 年龄
    "NumberOfTime30-59DaysPastDueNotWorse": "dpd30_59",   # 过去两年逾期 30–59 天的次数
    "DebtRatio": "debt_ratio",                            # 负债比：每月债务支出 / 月收入
    "MonthlyIncome": "income",                            # 月收入
    "NumberOfOpenCreditLinesAndLoans": "open_lines",      # 未结清的贷款和信用卡数量
    "NumberOfTimes90DaysLate": "dpd90",                   # 逾期 90 天以上的次数
    "NumberRealEstateLoansOrLines": "re_loans",           # 房贷及房产抵押类贷款数量
    "NumberOfTime60-89DaysPastDueNotWorse": "dpd60_89",   # 逾期 60–89 天的次数
    "NumberOfDependents": "dependents",                   # 家庭供养人数
}
df = df.rename(columns=rename)

# 把数据写进一个内存里的 SQLite 数据库，之后就能用 SQL 查询 raw 这张表
con = sqlite3.connect(":memory:")
df.to_sql("raw", con, index=False)

# %% 2. 样本量和坏客户比例
print("【样本量与坏客户比例】")
print(q("""
    SELECT COUNT(*)            AS n,
           SUM(bad)            AS n_bad,
           ROUND(AVG(bad), 4)  AS bad_rate
    FROM raw
"""), "\n")

# %% 3. 缺失情况（COUNT(列名) 不统计空值，所以 COUNT(*) - COUNT(列名) 就是缺失数）
print("【缺失值】")
print(q("""
    SELECT COUNT(*) - COUNT(income)                         AS income_missing,
           ROUND(1 - COUNT(income) * 1.0 / COUNT(*), 4)     AS income_missing_pct,
           COUNT(*) - COUNT(dependents)                     AS dependents_missing,
           ROUND(1 - COUNT(dependents) * 1.0 / COUNT(*), 4) AS dependents_missing_pct
    FROM raw
"""), "\n")
# 用 pandas 把所有列再核对一遍，确认其他列没有缺失
print(df.isna().sum(), "\n")

# %% 4. 描述统计：重点看最小值、最大值和 1% / 99% 分位数
print("【描述统计】")
print(df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.round(2), "\n")

# %% 5. 逾期次数这几列都有哪些取值？
print("【逾期次数的取值分布】")
for col in ["dpd30_59", "dpd60_89", "dpd90"]:
    print(f"-- {col}")
    print(q(f"""
        SELECT {col} AS value, COUNT(*) AS n
        FROM raw
        GROUP BY {col}
        ORDER BY {col}
    """).to_string(index=False), "\n")

# %% 6. 几个可疑取值各有多少条
print("【可疑取值计数】")
print(q("""
    SELECT SUM(CASE WHEN age = 0         THEN 1 ELSE 0 END) AS age_is_0,
           SUM(CASE WHEN revol_util > 1  THEN 1 ELSE 0 END) AS revol_util_gt_1,
           SUM(CASE WHEN revol_util > 10 THEN 1 ELSE 0 END) AS revol_util_gt_10,
           SUM(CASE WHEN income IS NULL AND debt_ratio > 10 THEN 1 ELSE 0 END)
                                                       AS no_income_and_debt_ratio_gt_10
    FROM raw
"""), "\n")

# %% 7. 不同年龄段的坏客户比例（age 是整数，age / 10 在 SQLite 里是整除）
print("【年龄段 × 坏客户比例】")
print(q("""
    SELECT (age / 10) * 10     AS age_band,
           COUNT(*)            AS n,
           ROUND(AVG(bad), 4)  AS bad_rate
    FROM raw
    GROUP BY age_band
    ORDER BY age_band
""").to_string(index=False), "\n")

# %% 8. 过去有没有逾期记录，坏客户比例差多少？
print("【过去是否逾期 × 坏客户比例】")
print(q("""
    SELECT CASE WHEN dpd30_59 + dpd60_89 + dpd90 > 0 THEN '有逾期记录'
                ELSE '无逾期记录' END   AS history,
           COUNT(*)                     AS n,
           ROUND(AVG(bad), 4)           AS bad_rate
    FROM raw
    GROUP BY history
""").to_string(index=False))
