
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

## Families (Simulation)
1. 村里的家庭都是有1，2，3个孩子的家庭。去学校random问100个学生，50个说家里1个小孩，20个说2个小孩，30个说3个小孩。问所有家庭里, 1孩家庭的概率。code出来。会围绕你写的code follow up一些问题。
2. 第一个就是那个一孩，二孩，三孩的问题，解出来之后问如何求confidence interval，我说就是simulation，不断地从pool里抽100个孩子。这个地方稍微有一点沟通和理解的问题，就是他最后告诉抽出来的数据应该是[ 1, 1, 1, ..., 2, 2,..., 3,...,3]这样，我以为是直接得到百分比，不过这一点也在沟通中得到解释，都互相理解了。

## 概率+粒子simulation 

2025
面试上来，彼此介绍一下，就开始出题，题目是个纯概率题，大概是一种粒子有个半衰期，问一百个粒子一定天数以后还存在没有衰变的粒子的概率。其实不太难，这基本是国内中学概率题的水平，不过太久没接触过此类问题，答得磕磕绊绊，中间还犯了点错，在面试官的提示下才完成。然后coding是对上面问题做一个simulation来检验理论结果, 要现场share screen写code并且run, 又是很久没code 的类型，做的同样磕磕绊绊。


考到了同一题，还有一些follow up：
- 怎么算“还存在没有衰变的粒子的概率”的confidence interval - binomial公式或者simulation. 
- 比如给了particle decay每天的outcome怎么样estimate衰变概率 - 用weighted avg，weight by size-我忘了这个的公式，在提示下磕磕绊绊的算出来了


2025
- 原子半衰期為1天, 問100個原子在10天後剩下存活的機率
- code for simulation
- follow up: how to improve TC and why it works? leverage numpy size argument, vector stored in memory closely therefore faster to fetch





