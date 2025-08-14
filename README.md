# Robust Exp

## 测试Correctness of Robust Pricing Model
### 测试1
- <font color="red"> 我们在main.ipynb中测试发现，若只用一个指数锥</font>
$\rho_n \geq \rho_0 \exp(a_u^\top u + b_u^\top p)$
- <font color="red"> 结果是采用ETO的价格，RO模型得到的目标函数值不等于ETO模型的目标函数值</font>
- <font color="red">因此，似乎需要考虑两个指数锥</font>$\rho_n \geq \rho_0 \exp(a_u^\top u + b_u^\top p)$ 和$\rho_n \leq \rho_0 \exp(a_u^\top u + b_u^\top p)$

### 测试2
- Robust_Correctness.ipynb: 测试ETO和RO(Uncertainty Set非常小的时候)的结果是否一致
```
- 非对角线元素是负的情况下，RO 可能存在和ETO结果不一样的情况，其他都是一致的
- 非对角线元素在所有情况下，Two_side RO 都存在和ETO结果不一致的情况
```

## 测试估计方法
- Estimate_Test.ipynb:测试估计参数的方法是否有效


## 仿真实验
- Main.ipynb