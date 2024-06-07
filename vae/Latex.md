#### 角标位置
当使用行内数学环境\$…\$时，由于采用的是inline mode模式，上下标默认是出现在右上和右下。

当使用行间数学环境\$\$…\$\$时，由于采用的是interline mode模式，上下标默认是出现在正上方正下方。

当在行内使用上下标，\\limits_{i=1}^{2} 例如：$\mathop{M}\limits_{i=1}^2$ , 其中\\mathop{ }将一般符号转为数学符号

#### 矩阵
https://zhuanlan.zhihu.com/p/266267223
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


#### 等号对齐
$$
\begin{aligned}
KL(P||Q) &=\sum_x p(x) log \dfrac{p(x)}{q(x)} \\
&= \sum_x p(x) log(p(x)) - \sum_x p(x) log(q(x)) \\
&= - H[P] + H(P,Q)
\end{aligned}
$$

#### 左对齐
$A = 3\\$
$
=\left[
\begin{array}{ccc}
cov(x_1,x_1) & \cdots & cov(x_1,x_n)\\
\vdots & \ddots & \vdots\\
cov(x_1,x_n) & \cdots & cov(x_n,x_n)
\end{array}
\right]\\
$
$1+2= 3\\$

#### 点乘、叉乘、除以
点乘：a \cdot b  
叉乘：a \times b  
除以：a \div b  

#### 十字 标记
单十字：\dag  
双十字：\ddag

#### 无穷大
正无穷大： +\infty  
负无穷大： -\infty

#### 空格
$a\;b$

