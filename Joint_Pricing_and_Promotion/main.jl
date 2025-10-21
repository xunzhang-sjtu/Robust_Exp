using LinearAlgebra
using Distributions
using Optim
using Random
using StatsFuns
using JuMP
using MosekTools
using StatsBase
using SparseArrays # 可选，用于处理稀疏性（如果数据量很大）
using FileIO
using JLD2
using Plots
using LaTeXStrings
using DataFrames, Colors
using StatsPlots   # 提供 boxplot，基于 Plots

include("Params_PP.jl")
include("Data_Generation_PP.jl")
include("Estimation_PP.jl")
include("Evaluation_PP.jl")
# # include("Plot_Figures.jl")
include("Implement_All_Methods_PP.jl")
include("Models_PP.jl")


function Generate_Input_this(S_train, N, N_u, K, offdiag_sign,max_offdiag,P_bar,is_original_setting)
    Input_Data_this = Dict()
    A_true, B_true = Generate_Coef(N_u, N, max_offdiag, offdiag_sign,is_original_setting);
    P_train,PM_train,choice_train,PM_train_extend = Generate_Data(N,S_train,A_true,B_true,P_bar);

    Input_Data_this["A_true"] = A_true;
    Input_Data_this["B_true"] = B_true;
    Input_Data_this["P_dag"] = round.(rand(N, K) .* P_bar; digits=2);
    Input_Data_this["P_train"] = P_train;
    Input_Data_this["PM_train_extend"] = PM_train_extend;
    Input_Data_this["choice_train"] = choice_train;

    A_hat,B_hat = Estimate_MNL_Para(PM_train_extend, P_train, choice_train,S_train, N);

    Input_Data_this["A_hat"] = A_hat
    Input_Data_this["B_hat"] = B_hat
    return Input_Data_this
end

function Get_Input_Data(Input_Data_this)
    A_true = Input_Data_this["A_true"]
    B_true = Input_Data_this["B_true"]
    P_dag = Input_Data_this["P_dag"]
    P_train = Input_Data_this["P_train"]
    PM_train_extend = Input_Data_this["PM_train_extend"]
    choice_train = Input_Data_this["choice_train"]
    A_hat = Input_Data_this["A_hat"]
    B_hat = Input_Data_this["B_hat"]
    return A_true,B_true,P_dag,P_train,PM_train_extend,choice_train,A_hat,B_hat
end

function solve_ETO_this(N,N_u,K,A,B,A_true, B_true,P_dag,Time_Limit)
    RST_this = Dict()
    status_this = "NotDefined"
    obj_,X_,Promo_, time_,status_ = Solve_ETO(N,N_u,K,A, B,P_dag,Time_Limit)
    if status_ != "OPTIMAL"
        status_this = status_
    else
        status_this = status_
        rev_, price_ = compute_oof(X_, A_true, B_true, vcat(Promo_,1), P_dag)
        RST_this["obj"] = obj_
        RST_this["price"] = price_
        RST_this["promo"] = Promo_
        RST_this["time"] = time_
        RST_this["Rev"] = rev_
        RST_this["status"] = status_
    end
    return RST_this,status_this
end

function solve_RO_this(N,N_u,K,A_hat,B_hat,A_true, B_true,P_dag,psi_lb,psi_ub,phi_lb,phi_ub,gamma_list,dual_norm,Time_Limit)
    RST_this = Dict()
    status_this = "NotDefined"
    for gamma in gamma_list
        obj_RO,X_RO,Promo_RO, time_RO,status_RO = Solve_RO(N,N_u,K,A_hat,B_hat,P_dag,psi_lb,psi_ub,phi_lb,phi_ub,gamma * ones(N),dual_norm,Time_Limit)
        println("gamma=$(gamma),status:",status_RO,",time=$time_RO")
        if status_RO != "OPTIMAL"
            status_this = status_RO
            break
        else
            status_this = status_RO
            rev_RO, price_RO = compute_oof(X_RO, A_true, B_true, vcat(Promo_RO,1), P_dag)
            RST_this["obj_gamma=$(gamma)"] = obj_RO
            RST_this["price_gamma=$(gamma)"] = price_RO
            RST_this["promo_gamma=$(gamma)"] = Promo_RO
            RST_this["time_gamma=$(gamma)"] = time_RO
            RST_this["Rev_gamma=$(gamma)"] = rev_RO
            RST_this["status"] = status_this
        end
    end
    return RST_this,status_this
end

Params = get_default_params_PP()
seed = Params["seed"]
N = Params["N"]
N_u = Params["N_u"] 
K = Params["K"] 
S_train = Params["S_train"]
S_test = Params["S_test"]
P_bar = Params["P_bar"]
iterations = Params["iterations"]
offdiag_sign = Params["offdiag_sign"]
max_offdiag = Params["max_offdiag"]
Time_Limit = Params["Time_Limit"]
gamma_list = Params["gamma_list"]
dual_norm = Params["dual_norm"]
psi_lb = Params["psi_lb"]
psi_ub = Params["psi_ub"]
phi_lb = Params["phi_lb"]
phi_ub = Params["phi_ub"]

is_original_setting = true
Random.seed!(seed)

project_dir = "Joint_Pricing_and_Promotion/"
current_dir = pwd()
parent_dir = dirname(current_dir)
grand_pa_dir = dirname(parent_dir)
data_dir = string(grand_pa_dir, "/Data/")
if !isdir(data_dir)
    mkpath(data_dir)
end
if is_original_setting
    sub_file_name = "Test_MS_2024_N=$(N)_N_u=$(N_u)_K=$(K)_S_train=$(S_train)_Seed=$seed/"
else
    sub_file_name = "N=$(N)_N_u=$(N_u)_K=$(K)_S_train=$(S_train)_offdiag_sign=$(offdiag_sign)_max_offdiag=$(max_offdiag)/"
end
this_data_file = string(data_dir,project_dir,sub_file_name)
if !isdir(this_data_file)
    mkpath(this_data_file)
end
println(this_data_file)
save(string(this_data_file, "Params.jld2"), Params);

Result_All = Dict()
Data_all = Dict()
global d_it = 1
global iter = 1
while iter <= iterations
    t_start = time()  # 记录开始时间
    Input_Data_this = Generate_Input_this(S_train, N, N_u, K, offdiag_sign,max_offdiag,P_bar,is_original_setting)
    Data_all["d_iter=$(d_it)"] = Input_Data_this
    global d_it = d_it + 1
    save(string(this_data_file, "Data_all.jld2"), Data_all);

    t_data = time()  # 记录结束时间
    println("******* Time for data d_iter $(d_it-1): ", round(t_data - t_start, digits=2), " seconds *********")

    A_true,B_true,P_dag,P_train,PM_train_extend,choice_train,A_hat,B_hat = Get_Input_Data(Input_Data_this)
    if any(isnan, A_hat) || any(isnan, B_hat)
        println("Estimate contains NaN values")
        continue
    end
    Result_All["Input_Data_iter=$(iter)"] = Input_Data_this

    RST_Oracle,status_Oracle = solve_ETO_this(N,N_u,K,A_true,B_true,A_true, B_true,P_dag,Time_Limit)
    println("Oracle: status = ",status_Oracle,",time=",RST_Oracle["time"])
    if status_Oracle != "OPTIMAL"
        println("Oracle did not approach the optimal solution")
        continue
    end
    Result_All["RST_Oracle_iter=$(iter)"] = RST_Oracle

    t_true = time()  # 记录结束时间
    println("True d_iter $(d_it-1): Time=", round(t_true - t_data, digits=2), ",obj=", RST_Oracle["obj"])



    RST_ETO,status_ETO = solve_ETO_this(N,N_u,K,A_hat,B_hat,A_true, B_true,P_dag,Time_Limit)
    println("ETO: status = ",status_ETO,",time=",RST_ETO["time"])
    if status_ETO != "OPTIMAL"
        println("ETO did not approach the optimal solution")
        continue
    end
    Result_All["RST_ETO_iter=$(iter)"] = RST_ETO
    
    t_ETO = time()  # 记录结束时间
    println("ETO d_iter $(d_it-1): time=", round(t_ETO - t_true, digits=2), ",obj=", RST_ETO["obj"])



    obj_RO,X_RO,Promo_RO, time_RO,status_RO = Solve_RO(N,N_u,K,A_hat,B_hat,P_dag,psi_lb,psi_ub,phi_lb,phi_ub,0.0 * ones(N),dual_norm,Time_Limit)
    if status_RO != "OPTIMAL"
        println("RO did not approach the optimal solution")
        continue
    end
    rev_RO, price_RO = compute_oof(X_RO, A_true, B_true, vcat(Promo_RO,1), P_dag)
    if abs(rev_RO - RST_ETO["Rev"]) >= 0.001
        println("ETO rev is not equivalent to RO rev")
        continue
    end
    t_RO1 = time()  # 记录结束时间
    println("RO1 d_iter $(d_it-1): time=", round(t_RO1 - t_ETO, digits=2), ",obj=", obj_RO)



    RST_RO,status_RO = solve_RO_this(N,N_u,K,A_hat,B_hat,A_true, B_true,P_dag,psi_lb,psi_ub,phi_lb,phi_ub,gamma_list,dual_norm,Time_Limit)
    # println("RO status = ",status_RO)
    if status_RO != "OPTIMAL"
        println("RO hello world")
        continue
    end
    # Result_All["Input_Data_iter=$(iter)"] = Input_Data_this
    # Result_All["RST_Oracle_iter=$(iter)"] = RST_Oracle
    # Result_All["RST_ETO_iter=$(iter)"] = RST_ETO
    Result_All["RST_RO_iter=$(iter)"] = RST_RO
    save(string(this_data_file, "Result_All.jld2"), Result_All);
    t_RO = time()  # 记录结束时间
    println("RO d_iter $(d_it-1): ", round(t_RO - t_RO1, digits=2), " seconds *********")
    t_end = time()  # 记录结束时间
    println("******* Time for iter $iter: ", round(t_end - t_start, digits=2), " seconds *********")
    println("--------------------------------------------------")
    global iter = iter + 1
end
save(string(this_data_file, "Result_All.jld2"), Result_All);