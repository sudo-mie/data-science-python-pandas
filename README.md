
## Linear Regression

#### TODO 2025 
y=x+e，做OLS, regress x on y， 问y的系数？


一个典型的causal题目，让求Y～X的regression的coefficient，然后求反过来X～Y的regression的系数，最后让解释为什么不一样，这是典型的reverse causality。很快就进入第三题，一个regression model里面系数都不显著，但是performance很好是什么原因。因为collinearity。如何解决？LASSO regularization。


#### TODO-2 2025 - 
We have random samples for 3 variables X,Y and Z, where X and Z's sample correlation coefficient is 0. We fit a sample linear regression of Y on X with intercept, and the coefficient of the determination R^2 = 0. If we fit a linear regression of X on Y with intercept, what will the coefficient of determination R^2 be?


 
Revised Question: "We have random samples for variables X, Y, and Z. The sample correlation between X and Z is 0. If we fit a simple linear regression of Y on X, and then a multiple linear regression of Y on both X and Z, how does the R^2 change?"
感觉题目改成这样比较合理






## Old Toss Die

#### 2024 
一个dice投出所有面，需要投多少次

#### 2023.11
1a. 给你一个n面随机 骰子，请问丢多少次骰子的所有面都会出现？
会给你一个结果为1到n的随机函数，请你写一个函数得到所有面出现的次数


1b. 请问如何编程得出这个次数的统计参数：均值，中位数，STD。


1c. 请用数学方法得出这个分布的期待。
提示，撒出一个数概率为p, 那么一直撒直到得到这个数的概率是1/p ..
这题反复和面试官探讨后觉得是 n(1/n + 1/(n-1) + 1/(n-2) + ... + 1/1), 不知道是否正确

#### 2022
Roll a fair die n time, let M_n be the maximum of all the rolls. What's the probability of M_n=r, for each r=1,2,...,6?

## Coding - Pandas

#### 1. Replace the nulls with the average for its own category in the table table.

```
df['value'] = df['value'].fillna(df.groupby('category')['value'].transform('mean'))

# null就是用字符串的‘null’来表示，所以在把这一列转换成float，同时把‘null’转换成np.nan
df['value'] = pd.to_numeric(df['value'].replace('null', np.nan))
```

#### 2. Return the values that fall outside one standard deviation of mean.
```
mean = df['value'].mean()
std = df['value'].std()
# Create a boolean mask for values outside ±1 std
mask = (df['value'] < mean - std) | (df['value'] > mean + std)
outliers = df.loc[mask, 'value']

# OR: combine to 1-liner
outliers = df.loc[(df['value'] - df['value'].mean()).abs() > df['value'].std(), 'value']

```




