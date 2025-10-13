# R1

一个是security level，一个是company level

### inspect data

```
df.head()
```

### merge data

一个是security level，一个是company level 
(直接merge就行 / 根据id join)


```
pd.merge(df1, df2, on='ID', how='left')

pd.merge(table1,table2, how='inner’', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=None, indicator=False, validate=None)
```

#### 可以查看duplicates （实际没有）

```
	dupes = df[df.duplicated(subset=["security_id", "time"], keep=False)]
dupes.sort_values(["security_id", "time"])
# Select rows with no duplicated lat/lon
df = insurance[~insurance.duplicated(subset=['lat', 'lon'], keep=False)]
```

### check date range

主要check时间 (price的time在sbology的有效期内。sbology是有一个时间range这种？)

```
company_df["start_date"] = pd.to_datetime(company_df["start_date"])
mask_in_range = (df["time"] >= df["start_date"]) & (df["time"] <= df["end_date"])
violations = merged.loc[~mask_in_range]
```


### column A + B，然后Groupby

多算出来俩新的column，groupby，求sum
算per company or per secuirty的metrics

```
per_company_sum =  df.groupby("company_id", as_index=False)[["dollar_volume","market_cap"]]
         	  .sum()
        	 .rename(columns={"dollar_volume":"sum_dollar_volume","market_cap":"sum_market_cap"})

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
         .sum()
         .rename(columns={"w_mcap":"weighted_monthly_avg_mcap"})
)

```


### Sort选top 5
	求出来每个月market cap最多的5个公司
	per company的钱的排序

Q: subset the top 5 company

```

# top 5 by month
df_sorted = df_weighted.sort_values(by=['year', 'month','weighted_mkt_cap'], ascending=[True, True, False])

df_top = df_sorted.groupby(['year', 'month']).head(5)
```

# R2

data是portfolio，有好几个strategy

### Plot
- 画图（类似backtest, 每个strategy的return的图）
- 普通的图，看某一个column长什么样子 df.plot就行

假设table是
```
date | strategy | portfolio_value


pivoted_df = df.pivot(index="date", columns="strategy", values="portfolio_value")
pivoted_df.plot(title="Portfolio Value by Strategy", figsize=(10,5))
plt.ylabel("Portfolio Value ($)")
plt.show()
```


### 计算

- group by, count unique element
- 需要知道，max drawdown / sharpe ratio的概念和算法
- 一年多少个trading day （除一下）

```
# group by
df.groupby('column_name').filter(lambda x: x['column_name'].count() > 10)
df.groupby("strategy")["date"].nunique()

```


#### daily return

```
df = df.sort_values(["strategy", "date"])
df["daily_return"] = df.groupby("strategy")["portfolio_value"].pct_change()
```



#### annualized return

```
trading_days_per_year = 252

# 计算总收益率
summary = df.groupby("strategy").agg(
    start_value = ("portfolio_value", "first"),
    end_value = ("portfolio_value", "last"),
    n_days = ("date", "nunique")
)

summary["total_return"] = summary["end_value"]/summary["start_value"] - 1
summary["annualized_return"] = (1 + summary["total_return"])**(trading_days_per_year / summary["n_days"]) - 1
```


#### max drawdown

```
def max_drawdown(series):
    # series: cumulative portfolio value
    roll_max = series.cummax()
    drawdown = 1 - series / roll_max
    return drawdown.max()

# 应用于每个策略
mdd = df.groupby("strategy")["portfolio_value"].apply(max_drawdown)
summary["max_drawdown"] = mdd
```

#### sharpe ratio

```
def sharpe_ratio(x):
    return (x.mean() / x.std()) * (252**0.5)

sharpe = df.groupby("strategy")["daily_return"].apply(sharpe_ratio)
summary["sharpe_ratio"] = sharpe
```

#### risk (volatitliy, downside sharpe ratio - 亏钱的天的STD DEVIATION)





## bonus question

- 关于sharpe/drawdown的conceptual，跟coding无关
- risk，portfolio的理解






