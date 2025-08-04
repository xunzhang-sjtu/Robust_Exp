using LinearAlgebra
using Distributions
using Optim
using Random
using StatsFuns
using JuMP
using MosekTools
include("ETO.jl")
include("RO.jl")
include("Data.jl")
include("Estimate.jl")
include("Performance.jl")
Random.seed!(2)


N_u = 1
N = 2
S = 100
S_test = 10000
K = 10   # 每个产品的选择项数量

max_offdiag = 1
offdiag_sign = "mix"


A_true, B_true = Generate_Coef(N_u, N,max_offdiag,offdiag_sign);


U_train, P_train = Generate_Feat_Data(N_u, N, S);
U_test, P_test = Generate_Feat_Data(N_u, N, S_test);
probs = Calculate_Prob(S,N, U_train, P_train, A_true, B_true);
choices = Calculate_Choice(S,N,probs);

result = Estimate_MNL_Para(N, N_u, N, U_train, P_train, choices)
