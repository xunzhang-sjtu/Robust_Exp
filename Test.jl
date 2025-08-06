using LinearAlgebra
using Distributions
using Optim
using Random
using StatsFuns
using DataFrames
using StatsBase
include("Estimate.jl")
# 设置随机种子
Random.seed!(1)


# 模型设置
N = 3           # 选项数（不含baseline）
d_u = 1         # u 的维度
d_p = N         # p 的维度
S = 5000         # 样本数量


a_star = [rand(Uniform(-1, 1), d_u) for n in 1:N]
b_star = [rand(Uniform(-1, 1), d_p) for n in 1:N]
a_star = [round.(a; digits=2) for a in a_star]
b_star = [round.(b; digits=2) for b in b_star]
println("a_star: ", a_star) 
println("b_star: ", b_star) 

# Step 2: 生成样本 (u, p)，标准正态分布
U_train = [randn(d_u) for s in 1:S];
P_train = [randn(d_p) for s in 1:S];

Estimate_MNL_Para(U_train, P_train, S, N, d_u, d_p, a_star, b_star)