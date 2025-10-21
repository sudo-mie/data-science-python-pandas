
## Linear Regression

#### 2025 
y=x+e，做OLS, regress x on y， 问y的系数？


#### 2025 - [RS] 
一个典型的causal题目，让求Y～X的regression的coefficient，然后求反过来X～Y的regression的系数，最后让解释为什么不一样，这是典型的reverse causality。很快就进入第三题，一个regression model里面系数都不显著，但是performance很好是什么原因。因为collinearity。如何解决？LASSO regularization。


#### 2025 - 
We have random samples for 3 variables X,Y and Z, where X and Z's sample correlation coefficient is 0. We fit a sample linear regression of Y on X with intercept, and the coefficient of the determination R^2 = 0. If we fit a linear regression of X on Y with intercept, what will the coefficient of determination R^2 be?

Let X and Y be random variables with E(X) = 4, E(Y) = 6, VAR(X) = 5, VAR(Y) = 3, COV(X, Y) = 0. We generate i.i.d samples from X and Y: Xi, Yi, where i=1, 2, …, n. Suppose we do a linear regression, Y_i = a + bX_i + e_i and obtain a_hat and b_hat. What's your best estimate of E[a_hat] and E[b_hat] ?
 
Revised Question: "We have random samples for variables X, Y, and Z. The sample correlation between X and Z is 0. If we fit a simple linear regression of Y on X, and then a multiple linear regression of Y on both X and Z, how does the R^2 change?"
感觉题目改成这样比较合理

## Toss Die 
2025.1 -  
Assume you have a fair die, and toss it 6 times, each time you write down the number on the top side. What is the probability that the sum of these 6 numbers is a multiple of 6?


## Toss Coin
2025 - [RS] 
一个quant/probability的题: What is the expected number of streaks if we toss the coin 1000 times? Definition of streak: Each time the outcome switches from heads to tails or tails to heads, a new streak begins. That means a streak starts at: the very first toss, or every time the coin result differs from the previous one.

答案是500.5
follow up:就是如果这个不是一个fair coin，like head with probability 5/7，结果会是什么，其实就是把计算的数据改一下就行，公式都不用变



