# R1

一个是security level，一个是company level

### Pull Data
```
url = "https://example.com/data.csv" # "https://example.com/data.csv.zip"
df = pd.read_csv(url)

df = pd.read_excel("https://example.com/data.xlsx", sheet_name="Sheet1")
```
### inspect data

```
df.head()
df.info()
```
#### 可以查看duplicates/missing data （实际没有）
```
	dupes = df[df.duplicated(subset=["security_id", "time"], keep=False)]
dupes.sort_values(["security_id", "time"])
# Select rows with no duplicated lat/lon
df = insurance[~insurance.duplicated(subset=['lat', 'lon'], keep=False)]
df.isna().sum()
```
### merge data
- 一个是security level，一个是company level 
- 直接merge就行
- 根据id join

```
pd.merge(df1, df2, on='ID', how='left')

pd.merge(table1,table2, how='inner’', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=None, indicator=False, validate=None)

# 哪些 security 行没有任何公司匹配
no_owner = merged[merged["company_id"].isna()]
```
### check date range
主要时间，price的time在sbology的有效期内。sbology是有一个时间range这种？

TODO: 
- for date, if year, month, day are in separated columns?

```
company_df["start_date"] = pd.to_datetime(company_df["start_date"])
mask_in_range = (df["time"] >= df["start_date"]) & (df["time"] <= df["end_date"])
violations = merged.loc[~mask_in_range]

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day


```



### Calculation: column A + B，然后Groupby
- 多算出来俩新的column，groupby，求sum
- 算per company or per secuirty的metrics
 dollar_volume, market_cap


company_df
| year | month | company_id | security_id | volume | market_cap |


security_df
| security_id | start_date | end_date | price |


|year|month|company_id|security_id|volume|start_date| end_date | price |

=> 
|year|month|company_id|security_id|volume|price|
=> 
|year|month|company_id|security_id|volume|price|dollar_volume

dollar_volume = volume * price



```
df["dollar_volume"] = df["volume"] * df["price"]

df_company['total_dollar_volume'] = df_company.groupby(['year', 'month','company_id'])['dollar_volume'].transform('sum')

#rename column names after sum
.rename(columns={"position_value": "portfolio_value"})

# groupby + apply: flexible
def add_group_max(df_group):
    df_group['group_max_score'] = df_group['score'].max()
    return df_group
df = df.groupby('group_column').apply(add_group_max).reset_index(drop=True)
# aka
employee.groupby('departmentId').apply(lambda x: x[x['salary'] == x['salary'].max()]).reset_index(drop=True)


# groupby + transform: row-wise, same shape
df_grouped['score_standardized'] = df_grouped.groupby('subject')['score'].transform(lambda x: (x - x.mean()) / x.std())

# groupby + filter: keep or remove entire group
# return a df with only the groups that have more than 10 rows
df.groupby('column_name').filter(lambda x: x['column_name'].count() > 10)

```

#### 会需要算cumulative （sum, prod)

```
df["cum_sum"] = df["value"].cumsum()
df["cum_prod"] = df["value"].cumprod()

#need to sort first for timeseries
df = df.sort_values(["company","day"])

#with groupby
df["cum_revenue"] = df.groupby("company")["revenue"].cumsum()

#multi columns
df[["cum_rev", "cum_prod"]] = (
    df.groupby("company")[["revenue"]]
      .transform(lambda x: pd.concat([x.cumsum(), x.cumprod()], axis=1))
)

#with shift
df["prev_cum"] = df.groupby("company")["revenue"].cumsum().shift()
df["growth_rate"] = (df["cum_revenue"] / df["prev_cum"]) - 1
```


### Weighted average

Q: weighted monthly average market cap per company
The way C calculates and populates the monthly total volume (to be used the denominator for weights) for each company is pretty clean

```

df_company['total_dollar_volume'] = df_company.groupby(['year', 'month','company_id'])['dollar_volume'].transform('sum')

df_company['weight'] = df_company['dollar_volume']  / df_company['total_dollar_volume'] 

df_company["w_mcap"] = df_company["weight"] * df_company["market_cap"]

monthly_weighted_mcap = (
    df_company.groupby(["year","month","company_id"], as_index=False)["w_mcap"]
         .sum() # keep only one row per group
         .rename(columns={"w_mcap":"weighted_mkt_cap"})
)

output: | year | month | company_id | weighted_mkt_cap |

```
### Sort选top 5
	- 求出来每个月market cap最多的5个公司
	- per company的钱的排序

Q: subset the top 5 company

# top 5 by month
df_sorted = df_weighted.sort_values(by=['year', 'month','weighted_mkt_cap'], ascending=[True, True, False])

df_top = df_sorted.groupby(['year', 'month']).head(5)


# R2
portfolio，好几个strategy

假设table是
date            | strategyA | strategyB
2025.01.01 | 98000      | 99000

OR: 
date            | strategy | portfolio_value
2025.01.01 |   A          | 98000
2025.01.01 |   B          | 99000 

### Load data
```
values = pd.read_csv("your_values.csv", parse_dates=["date"], index_col="date")
df.info()
df.head() 
df.shape
df["date"] = pd.to_datetime(df["date"])  # convert to datetime type
df = df.set_index("date")                # make it the index

# check missing value
df.isna().sum()


# check date is continuous
expected = pd.bdate_range(df.index.min(), df.index.max())  # business days
missing = expected.difference(df.index)
```
### Plot

```
# make sure it is sorted
df = df.sort_values(["date"]) # df = df.sort_values(["date",”strategy”])

ax = df.plot(figsize=(10,5))
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value")
plt.show()

# Convert 2 to 1
df_pivot = df.pivot(index="date", columns="strategy", values="portfolio_value")
df_pivot.columns.name = None


df_pivot.plot(title="Portfolio Value by Strategy", figsize=(10,5))
plt.show()
```
### Calculation

#### count unique value
```
df["strategy"].nunique() / df.index.nunique()
df[["date", "strategy"]].drop_duplicates().shape[0]
```
#### Daily return
```
df = df.sort_values(["date"]) # df = df.sort_values(["date",”strategy”])

daily_ret = values.pct_change().fillna(0.0) # 1st row fill 0
df["daily_return"] = df.groupby("strategy")["portfolio_value"].pct_change()
```


#### combine results to df: 
```
combined = pd.concat([df_values.add_suffix("_value"),
                      cum_ret.add_suffix("_cumret")],
                     axis=1)
```

#### total return
```
total_return = df.iloc[-1] / df.iloc[0] - 1
```

#### Cumulative return
```
cum_ret = (1.0 + daily_ret).cumprod() - 1.0

# plot
ax = cum_ret.plot(figsize=(10,5))
ax.set_xlabel("Date"); 
ax.set_ylabel("Cumulative Return")
plt.show()
```
#### Annualized return
Check date range. If too short - flag that annualized return is not observed 

```
Option 1: Uses actual elapsed years between start & end 

years = (df.index[-1] - df.index[0]).days / 365.25 (or 365 if no leap years)
annualized_return = (df.iloc[-1] / df.iloc[0]) ** (1 / years) - 1


Option 2: Assume 252 trading days

annualized_return_scaled = (df.iloc[-1] / df.iloc[0]) ** (252 / len(df)) - 1

summary = df.groupby("strategy").agg(
    start_value = ("portfolio_value", "first"),
    end_value = ("portfolio_value", "last"),
    n_days = ("date", "nunique")
)
summary["total_return"] = summary["end_value"]/summary["start_value"] - 1
summary["annualized_return"] = (1 + summary["total_return"])**(252 / summary["n_days"]) - 1

Option 3: Scales by number of observations

trading_days_per_year = 252
def annualized_return_scaled(daily):
    total_return = (1 + daily).prod() - 1
    return (1 + total_return)**(trading_days_per_year / len(daily)) - 1
ann_ret = daily_ret.apply(annualized_return_scaled)

```


#### max drawdown

```
def max_drawdown(series):
    # series: cumulative portfolio value
    roll_max = series.cummax()
    drawdown = (roll_max - series) / roll_max
    return drawdown.max()

summary["max_drawdown"] = df.groupby("strategy")["portfolio_value"].apply(max_drawdown)

max_dd = df.apply(max_drawdown)
max_dd.head()

```

#### sharpe ratio
- assume risk-free rate ≈ 0。

```
# annualized sharpe ratio
def sharpe_ratio(x):
    return (x.mean() / x.std()) * (252**0.5)

sharpe = df.groupby("strategy")["daily_return"].apply(sharpe_ratio)
summary["sharpe_ratio"] = sharpe
sharpe = daily_ret.apply(sharpe_ratio)


# has RISK_FREE_ANNUAL_RATE
def sharpe_ratio(d: pd.Series, rf_annual=RISK_FREE_ANNUAL_RATE):
    daily_rf = (1 + rf_annual)**(1/252) - 1
    excess = d - daily_rf
    vol = d.std(ddof=1)
    return float(excess.mean() / vol * np.sqrt(252)) if vol != 0 else np.nan

```

#### annualized volatility
```
ann_vol = daily_ret.std(ddof=1) * np.sqrt(252)
```



#### downside sharpe ratio - 亏钱的天的STD DEVIATION
TODO

#### Any other portfolio metrics?

#### bonus question
- 关于sharpe的drawdown，conceptual, 跟coding无关
- risk，portfolio的理解



Dispersion risk: annualized volatility for noise/smoothness.


Path risk: max drawdown depth and duration (capital impairment + investor pain).


Loss-focused risk: downside deviation & downside Sharpe (penalize only bad tails).


Tail/regime risk: skew/kurtosis; stress windows (e.g., selloffs); rolling metrics.


Portfolio view: cross-correlations, risk contribution, turnover/leverage constraints.

