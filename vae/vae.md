### 最大似然估计  
似然函数如下：
$\mathcal{L}(\theta|x)=p(x|\theta)$  
更严格地，也可写成 $\mathcal{L}(\theta|x)=p(x;\theta)$

似然性（likelihood）与概率（possibility）同样可以表示事件发生的可能性大小，但是二者有着很大的区别：

概率 
 - 是在已知参数 $\theta$ 的情况下，发生观测结果 $x$ 可能性大小；  
  
似然性   
 - 则是从观测结果 $x$ 出发，分布函数的参数为$\theta$的可能性大小；

若 已知$x$ 未知$\theta$，对于两个参数 $\theta_1$,$\theta_2$有 $p(x|\theta_1)>p(x|\theta_2)$  

则 $\mathcal{L}(\theta_1|x)>\mathcal{L}(\theta_2|x)$

#### 最大似然估计  
最大似然估计方法（Maximum Likelihood Estimate，MLE）

最大似然估计的思想在于，对于给定的观测数据$x$，我们希望能从所有的参数
$\theta_1$,$\theta_2$,...,$\theta_{n}$ 中找出能最大概率生成观测数据的参数 $\theta^*$作为估计结果。

 $\mathcal{L}(\theta^*|x)\geq\mathcal{L}(\theta|x),\theta=\theta_1,...,\theta_n$

$p(x|\theta^*)\geq p(x|\theta)$  

最大化概率函数的参数即可：

$\theta^*= \mathop{argmax} \limits_\theta(p|\theta)$

![](https://pic2.zhimg.com/80/v2-2aa8f8a8ad7d454e9266c1bad5a3a83d_720w.webp)


#### 离散型随机变量的最大似然估计  

离散型随机变量$X$的分布律为$P\{X=x\}=p(x;\theta)$，设$X_1,...,X_n$为来自$X$的样本，$x_1,...,x_n$为相应的观察值，为待估参数。在参数$\theta$下，分布函数随机取到$x_1,...,x_n$的概率为$p(x|\theta)=\prod\limits_{i=1}^n p(x_i;\theta)$, 其中$\prod$是$\pi$的大写，表示累乘。  
通过似然函数 $\mathcal{L}(\theta|x)=p(x|\theta)=\prod\limits_{i=1}^n p(x_i;\theta)$   

此时 $\mathcal{L}(\theta|x)$ 是一个关于$\theta$的函数，寻找生成$x$的最大概率， 导数等于0时，取得极值：  
$\frac{d}{d\theta} L(\theta|x) = 0$   
因为$\prod\limits_{i=1}^n p(x_i;\theta)$是累乘形式，由复合函数的单调性，对原函数取对数：  
$\frac{d}{d\theta} ln L(\theta|x) = 1/L(\theta|x) \cdot \frac{d}{d\theta} L(\theta|x)  = 0$   

0-1分布(伯努利分布)
设随机变量X的分布律为 
X|0|1
:-----|:-----:|------:
P|1-p|p

![](https://pic2.zhimg.com/80/v2-b243cbfea33f0fa252eee4720d26018d_720w.webp)

这里$p$就是参数$\theta$

最大似然会缺点：数据量少的时候，会发生过拟合  
解决办法：加大数据量；先验的知识来纠偏, 贝叶斯的方法
$P(\theta|\mathcal{X})=\dfrac{P(\mathcal{X}|\theta)P(\theta)}{P(\mathcal{X})}$


### 蒙特卡洛方法(MC)
大数定律：如果统计数据足够大，那么事物出现的频率就能无限接近他的期望值。

如果$\{X_i\}_{i=1}^{n}$ 独立同分布，那么 $\dfrac{1}{n}\sum_{i=1}^{n}X_{i}\rightarrow \mathrm{E}(X)$

如果$\{X_i\}_{i=1}^{n}$ 独立同部分, 那么 $\{f(X_i)\}_{i=1}^{n}$ 也是独立同分布的，
且$\dfrac{1}{n}\sum_{i=1}^{n}f(X_{i})\rightarrow \mathrm{E}(f(X))$

示例：估计圆的面积

维度灾难：  
$n$ 维的球体的体积，$V_n=\dfrac{\pi^{\frac{n}{2}}R^{n}}{\Gamma(\frac{n}{2}+1)},~\Gamma(n+1)=n!$  
$n\rightarrow \infty, V_{n}\rightarrow 0,$  n维单位球体的体积趋向于0

### 概率论基础

#### 概率函数
概率密度函数(Probability Density Function, PDF)是描述随机变量在某个确定的取值点附近的可能性的函数。  
![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Boxplot_vs_PDF.svg/350px-Boxplot_vs_PDF.svg.png)

累积分布函数(Cumulative Distribution Function，CDF)是概率密度函数的积分。
![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Normal_Distribution_CDF.svg/300px-Normal_Distribution_CDF.svg.png)


概率质量函数（Probability Mass Function，PMF）是离散随机变量在各特定取值上的概率。
![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Discrete_probability_distrib.svg/220px-Discrete_probability_distrib.svg.png)

概率质量函数和概率密度函数的一个不同之处在于：概率质量函数是对离散随机变量定义的，本身代表该值的概率；概率密度函数本身不是概率，只有对连续随机变量的概率密度函数必须在某一个区间内被积分后才能产生出概率。


#### 链式法则
$P(A_1 \cap A_2 \cap \ldots \cap A_n) = P(A_1) \times P(A_2 \mid A_1) \times \ldots P(A_n \mid A_1 \cap \ldots \cap A_{n-1})$  
$P(X_{1}X_{2}\ldots X_n) = P(X_1)P(X_2|X_1)\ldots P(X_{n}|X_{<n})$

$P(A_1 \cap A_2)$ 表示事件$A_1$和$A_2$同时发生。
$P(X_{1}X_{2})$ 表示样本$X_1$和$X_2$同时出现。

#### 贝叶斯公式
https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E8%AE%BA#%E6%A6%82%E7%8E%87%E5%85%AC%E7%90%86  
贝叶斯公式用来描述两个条件概率之间的关系，比如 P(A|B) 和 P(B|A)。按照乘法法则，P(A∩B)=P(A)·P(B|A)=P(B)·P(A|B)，可以立刻导出贝叶斯定理  
$P(A|B)=\dfrac{P(B|A)P(A)}{P(B)}$   
“∣”读作given，即给定的意思。如 P(A∣B) 即 A given B 

#### 先验 后验 似然

- $先验=P(因)=P(\theta)$ 结果发生前, 就开始猜测(估计)原因， Prior
- $后验=P(因|果)=P(\theta|X)$ 已知结果，然后根据结果估计原因，Posterior
- $似然=P(果|因)=P(X|\theta)$ 先给定原因，根据原因来估计结果的概率分布，Likelihood
- $证据=P(果)=P(X)$ 出现结果的概率，```特别强调 这里的结果 反映的是在没有任何额外信息（即不知道结果）的情况下，出现结果的概率``` Evidence

$Posterior=\dfrac{Likelihood * Prior}{Evidence}$   

这里的因果只表示事件，不表示严格的因果推断。

有两个一模一样的箱子，箱子①里面有🍏🍏🍏和🍊；箱子②里面有🍏🍏和🍊🍊；  
(1) 随机选择一个箱子，从中摸出一个水果。选择箱子①的概率有多大？  
(2) 随机选择一个箱子，从中摸出一个水果发现是苹果。请问这个苹果来自箱子①的概率有多大？

事件发生顺序：先→选箱子，后→选水果。  
(1)  
$先验=选箱子=P(因)=P(\theta=①)=1/2$  
(2)  
已知结果求原因   
$后验=P(因|果)=P(\theta=①|X=🍏)=\dfrac{P(X=🍏|\theta=①) * P(\theta=①)}{P(X=🍏)}$  
现在已知 $P(\theta=①)=1/2, P(X=🍏|\theta=①)=3/4$  
求出$P(X=🍏)$即可得到$P(\theta=①|X=🍏)$  
在不考虑箱子的情况下摸到苹果的总概率，可以通过全概率公式计算：  
$$
\begin{aligned} 
P(X=🍏)&=P(X=🍏\cap\theta=①)+P(X=🍏\cap\theta=②) \\
&=P(X=🍏| \theta=①)P(\theta=①) + P(X=🍏|\theta=②)P(\theta=②) \\
&=5/8 \\
\\

P(\theta=①|X=🍏)&=\dfrac{3/4 * 1/2}{5/8}=3/5
\end{aligned}
$$


在VAE中，$Z$ 是隐变量，$X$ 是图像, 则有：  
$P(Z|X)=\dfrac{P(X|Z)P(Z)}{P(X)}$  
- 先验$P(Z)$ 
- 似然性$P(X|Z)$ 
- 后验$P(Z|X)$   
 
贝叶斯估计MAP，$P(\theta|\mathcal{X}) = \dfrac{P(\mathcal{X}|\theta)P(\theta)}{P(\mathcal{X})}$


### KL散度（Kullback-Leibler divergence） 
KL散度是两个概率分布P和Q差别的非对称性的度量。 KL散度是用来度量使用基于Q的分布来编码服从P的分布的样本所需的额外的平均比特数。典型情况下，P表示数据的真实分布，Q表示数据的理论分布。

KL散度的定义：  
$KL(P||Q)=\sum p(x) log \dfrac{p(x)}{q(x)}$


凸函数: 直观理解，凸函数的图像形如开口向上的杯
∪，而相反，凹函数则形如开口向下的帽 ∩。
二阶导数在区间上大于等于零，就称为凸函数。例如，$y=x^2$

![](https://upload.wikimedia.org/wikipedia/commons/4/4c/%E5%87%B8%E5%87%BD%E6%95%B0%E5%AE%9A%E4%B9%89.png)

概率论中, 有延森不等式: $f(E(X)) \leq E(f(X))$
这里把 $E(X)$想象成$\dfrac{x_1+x_2}{2}$, $E(f(X))$想象成$\dfrac{f(x_1)+f(x_2)}{2}$


吉布斯不等式:  
$$
\begin{aligned}
KL(P||Q) &=\sum_x p(x) log \dfrac{p(x)}{q(x)} \\
 &= - \sum_x p(x) log \dfrac{q(x)}{p(x)} \\
 &= E[-log \dfrac{q(x)}{p(x)}] \geq  -log[E(\dfrac{q(x)}{p(x)})] \\
 &= -log [\sum_x \bcancel{p(x)}  \dfrac{q(x)} { \bcancel{p(x)} }] \\
 &= -log [\sum_x q(x)] = - log1 = 0 \\
 KL(P||Q) \geq 0
\end{aligned}
$$

概率分布的熵 (H) 的定义是：  
$H[x]=-\sum_{x} p(x)log(p(x))$  

#### KL散度与交叉熵  
$$
\begin{aligned}
KL(P||Q) &=\sum_x p(x) log \dfrac{p(x)}{q(x)} \\
&= \sum_x p(x) log(p(x)) - \sum_x p(x) log(q(x)) \\
&= - H[P] + H(P,Q)
\end{aligned}
$$

H(P, Q) 称作P和Q的交叉熵（cross entropy）, KL散度不具备对称性，也就是说 P对于Q 的KL散度并不等于 Q 对于 P 的KL散度。

在信息论中，熵代表着信息量，H(P) 代表着基于 P 分布自身的编码长度，也就是最优的编码长度（最小字节数）。而H(P,Q) 则代表着用 P 的分布去近似 Q 分布的信息，自然需要更多的编码长度。并且两个分布差异越大，需要的编码长度越大。所以两个值相减是大于等于0的一个值，代表冗余的编码长度，也就是两个分布差异的程度。所以KL散度在信息论中还可以称为相对熵（relative entropy）。

为什么神经网络用没有用到KL散度，是因为 我们一般会 用交叉熵作为损失函数，在训练的过程中用 交叉熵 去逐渐靠近 真实熵。目的也是为了最小化KL散度。


### 高斯分布
[](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/1920px-Normal_Distribution_PDF.svg.png)

一维：$X\sim \mathcal{N}(\mu, \sigma^{2})$ , $p(x)=\dfrac{1}{\sqrt{2\pi\sigma^{2}}}\mathrm{exp}({-\dfrac{1}{2}(\dfrac{x-\mu}{\sigma})^{2}})$

$\mu$ 加权平均值(期望) $E(X)=\sum_{i}{p_i x_i}$  
$\sigma^2$ 方差(variance) $Var(X)=E[(X-\mu)^2]=E[X^2]-E[x]^2$  
【协方差 Covariance】用于度量两组数据的变量X和Y之间是否有线性相关性，=0不相关，>0正相关，<0负相关  
$cov(X,Y)=E[(X-E(X))(Y-E(Y))]= E[(X-\mu_X)(Y-\mu_Y)]$  
$cov(X,Y)=cov(Y,X)$  
$cov(aX,bY)=ab\;cov(Y,X)$  
【协方差矩阵】  
有 n 个随机变量组成一个 n维向量 $X=\{X_1,X_2,\cdots,X_n\}$
$$
\Sigma= cov(X,X^T) :=
\left[
\begin{array}{ccc}
cov(x_1,x_1) & \cdots & cov(x_1,x_n)\\
\vdots & \ddots & \vdots\\
cov(x_1,x_n) & \cdots & cov(x_n,x_n)
\end{array}
\right]
$$


【相关系数】用于度量两组数据的变量X和Y之间的线性相关的程度。它是两个变量的协方差与其标准差的乘积之比。  
$\rho_{X,Y}=\dfrac{cov(X,Y)}{\sigma_X \sigma_Y} = \dfrac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y}$  
$cov(X,Y)=\rho\;\sigma_X \sigma_Y$

皮尔逊相关系数的变化范围为-1到1。系数的值为1意味着X和 Y可以很好的由直线方程来描述，所有的数据点都很好的落在一条直线上，且 Y 随着 X 的增加而增加。系数的值为−1意味着所有的数据点都落在直线上，且 Y 随着 X 的增加而减少。系数的值为0意味着两个变量之间没有线性关系。  

特殊的，X自己和自己的协方差 $cov(X,X)=\sigma_X^2$，相关系数 $\rho_{X,X}=1$  
若 X 和 Y 相互独立(线性不相关)，$cov(X,Y)=0$，$\rho_{X,Y}=0$


【封闭性】数学中，若对某个集合的成员进行一种运算，生成的元素仍然是这个集合的成员，则该集合被称为在这个运算下闭合。


k维：$p(x)=\dfrac{1}{\sqrt{(2\pi)^{k}|\Sigma|}}\mathrm{exp}(-\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$


$\Sigma_{i,j} = cov(i,j) = \mathrm{E}[(X_i-\mu_i)(X_{j}-\mu_j)]$









参考文章  
贝叶斯深度学习及因果推断  
https://blog.csdn.net/weixin_42853410/article/details/127241293


Variational Mode Decomposition (变分模态分解)  
https://zhuanlan.zhihu.com/p/66898788

变分模态分解（VMD）  
https://blog.csdn.net/ximu__l/article/details/131031811

信号分解算法：小波包 EMD EEMD VMD LMD ACMD

生成模型（二） HVAE和VDM  
https://blog.csdn.net/FridaNN/article/details/131724696


Tutorial on Variational Autoencoders  
https://arxiv.org/pdf/1606.05908  

维基百科  
[理解协方差与协方差矩阵](https://blog.csdn.net/nstarLDS/article/details/104797269)  
[理解多元正态分布](https://blog.csdn.net/nstarLDS/article/details/104835010)  

[Latex矩阵](https://zhuanlan.zhihu.com/p/266267223)  

