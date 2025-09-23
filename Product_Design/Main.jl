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

include("Data_Generation_Product_Design.jl")
include("Estimation_Product_Design.jl")
include("Product_Design_Models.jl")
include("Evaluation_Product_Design.jl")

current_dir = pwd()
parent_dir = dirname(current_dir)
grand_pa_dir = dirname(parent_dir)
data_dir = string(grand_pa_dir, "/Data/Product_Design/20250922/")
if !isdir(data_dir)
    mkpath(data_dir)
end
println("Data directory: ", data_dir)


d = 5 # num of product feature
p = 10 # num of customer feature
n = 100 # num of training samples
n_test = 1
m = 5 # num of products
s = 10

lambda_list = [0.0001]
gamma_list = [0.0,0.0001,0.005,0.01,0.02,0.04,0.06,0.08,0.1]
psi_lb = -100
psi_ub = 100
phi_lb = -100
phi_ub = 100
instances = 100

Random.seed!(1)

Input_Data = Dict()
for ins in 1:instances
    data_dir_ins = string(data_dir, "instance=$ins/")
    # ****** Data ******
    theta_true, r_params = Generate_Wang_Qi_Max_True_Paras(d,p,s);
    X_train,Y_train,Z_train = Generate_Wang_Qi_Max_True_Data(d, p, n, m,theta_true);
    X_test,Y_test,Z_test = Generate_Wang_Qi_Max_True_Data(d, p, n_test, m,theta_true);

    Input_Data["theta_true_ins=$ins"] = theta_true
    Input_Data["r_params_ins=$ins"] = r_params
    Input_Data["X_train_ins=$ins"] = X_train
    Input_Data["Y_train_ins=$ins"] = Y_train
    Input_Data["Z_train_ins=$ins"] = Z_train
    Input_Data["X_test_ins=$ins"] = X_test
    Input_Data["Y_test_ins=$ins"] = Y_test
    Input_Data["Z_test_ins=$ins"] = Z_test
end
save(string(data_dir,"Input_Data.jld2"),Input_Data)


Estimate_Dict = Dict()
for ins in 1:instances
    data_dir_ins = string(data_dir, "instance=$ins/")
    X_train = Input_Data["X_train_ins=$ins"]
    Y_train = Input_Data["Y_train_ins=$ins"]
    Z_train = Input_Data["Z_train_ins=$ins"]
    for lambda in lambda_list
        alpha0_hat, alpha_hat, beta_hat, A_hat, opt_result = estimate_parameters(X_train,Y_train,Z_train,lambda, d, p, initial_theta=randn((d+1)*(p+1)) * 0.1);
        Estimate_Dict["alpha0_ins=$(ins)_lambda=$lambda"] = alpha0_hat
        Estimate_Dict["alpha_ins=$(ins)_lambda=$lambda"] = alpha_hat
        Estimate_Dict["beta_ins=$(ins)_lambda=$lambda"] = beta_hat
        Estimate_Dict["A_ins=$(ins)_lambda=$lambda"] = A_hat
        Estimate_Dict["opt_result_ins=$(ins)_lambda=$lambda"] = opt_result
    end
    println("************ data_dir_ins: ", data_dir_ins,"*************")
end
save(string(data_dir,"Estimate_Dict.jld2"),Estimate_Dict)

Result_True_Dict = Dict();
for ins in 1:instances
    data_dir_ins = string(data_dir, "instance=$ins/")
    theta_true = Input_Data["theta_true_ins=$ins"]
    r_params = Input_Data["r_params_ins=$ins"]
    X_train = Input_Data["X_train_ins=$ins"]
    Y_train = Input_Data["Y_train_ins=$ins"]
    Z_train = Input_Data["Z_train_ins=$ins"]
    X_test = Input_Data["X_test_ins=$ins"]
    Y_test = Input_Data["Y_test_ins=$ins"]
    Z_test = Input_Data["Z_test_ins=$ins"]
    r0 = r_params.r0;
    r = r_params.r;
    alp0_true = theta_true.alpha0;
    alp_true = theta_true.alpha;
    beta_true = theta_true.beta;
    A_true = theta_true.A;

    obj_True_list = zeros(n_test);
    x_True_list = zeros(n_test,d);
    time_True_list = zeros(n_test);
    profit_True_list = zeros(n_test);
    for i in 1:n_test
        z_input = Z_test[i,:];
        nu0 = alp0_true + beta_true' * z_input;
        nu = alp_true .+ A_true * z_input;
        # obj_True_list[i], x_True_list[i,:], time_True_list[i] = Product_Design_Ours_ETO(d,nu0, nu, r0, r, z_input);
        obj_True_list[i], x_True_list[i,:], time_True_list[i] = Product_Design_ETO(d,nu0, nu, r0, r, z_input);
        profit_True_list[i] = calculate_profit(alp0_true, alp_true, beta_true, A_true, r0, r, x_True_list[i,:], z_input)
        # if i % 20 == 1
        #     println("True: i=$i, obj=$(round(obj_True_list[i], digits=4)), profit=$(round(profit_True_list[i], digits=4)),x=$(round.(x_True_list[i,:], digits=4)), time=$(round(time_True_list[i],digits=4))")
        # end
    end
    println("************ data_dir_ins: ", data_dir_ins,"*************")
    Result_True_Dict["obj_ins=$ins"] = obj_True_list
    Result_True_Dict["sol_ins=$ins"] = x_True_list
    Result_True_Dict["time_ins=$ins"] = time_True_list
    Result_True_Dict["profit_ins=$ins"] = profit_True_list
end
save(string(data_dir,"Result_True.jld2"),Result_True_Dict) 


Result_ETO_Dict = Dict();
for ins in 1:instances
    data_dir_ins = string(data_dir, "instance=$ins/")
    theta_true = Input_Data["theta_true_ins=$ins"]
    r_params = Input_Data["r_params_ins=$ins"]
    X_train = Input_Data["X_train_ins=$ins"]
    Y_train = Input_Data["Y_train_ins=$ins"]
    Z_train = Input_Data["Z_train_ins=$ins"]
    X_test = Input_Data["X_test_ins=$ins"]
    Y_test = Input_Data["Y_test_ins=$ins"]
    Z_test = Input_Data["Z_test_ins=$ins"]
    r0 = r_params.r0;
    r = r_params.r;
    alp0_true = theta_true.alpha0;
    alp_true = theta_true.alpha;
    beta_true = theta_true.beta;
    A_true = theta_true.A;

    lambda_len = length(lambda_list)
    for l_index in 1:lambda_len
        lambda = lambda_list[l_index]
        alpha0_hat = Estimate_Dict["alpha0_ins=$(ins)_lambda=$lambda"]
        alpha_hat = Estimate_Dict["alpha_ins=$(ins)_lambda=$lambda"]
        beta_hat = Estimate_Dict["beta_ins=$(ins)_lambda=$lambda"]
        A_hat = Estimate_Dict["A_ins=$(ins)_lambda=$lambda"]

        obj_ETO_list = zeros(n_test);
        x_ETO_list = zeros(n_test,d);
        time_ETO_list = zeros(n_test);
        profit_ETO_list = zeros(n_test);
        for i in 1:n_test
            z_input = Z_test[i,:];
            nu0 = alpha0_hat + beta_hat' * z_input;
            nu = alpha_hat .+ A_hat * z_input;
            # obj_ETO_list[i], x_ETO_list[i,:], time_ETO_list[i] = Product_Design_Ours_ETO(d,nu0, nu, r0, r, z_input);
            obj_ETO_list[i], x_ETO_list[i,:], time_ETO_list[i] = Product_Design_ETO(d,nu0, nu, r0, r, z_input);
            profit_ETO_list[i] = calculate_profit(alp0_true, alp_true, beta_true, A_true, r0, r, x_ETO_list[i,:], z_input)
            # if i % 20 == 1
            #     println("ETO: lambda=$lambda,i=$i, 
            #     obj=$(round(obj_ETO_list[i], digits=4)), profit=$(round(profit_ETO_list[i], digits=4)),
            #     x=$(round.(x_ETO_list[i,:], digits=4)), time=$(round(time_ETO_list[i],digits=4))")
            # end
            println("ETO: ins=$ins,lambda=$lambda,i=$i, obj=$(round(obj_ETO_list[i], digits=4)), profit=$(round(profit_ETO_list[i], digits=4)),x=$(round.(x_ETO_list[i,:], digits=4)), time=$(round(time_ETO_list[i],digits=4))")
        end

        Result_ETO_Dict["obj_ins=$(ins)_lambda=$lambda"] = obj_ETO_list
        Result_ETO_Dict["sol_ins=$(ins)_lambda=$lambda"] = x_ETO_list
        Result_ETO_Dict["time_ins=$(ins)_lambda=$lambda"] = time_ETO_list
        Result_ETO_Dict["profit_ins=$(ins)_lambda=$lambda"] = profit_ETO_list
    end
    println("************ data_dir_ins: ", data_dir_ins,"*************")
end
save(string(data_dir,"Result_ETO.jld2"),Result_ETO_Dict) 


Result_RO_Dict = Dict();
for ins in 1:instances
    data_dir_ins = string(data_dir, "instance=$ins/")

    theta_true = Input_Data["theta_true_ins=$ins"]
    r_params = Input_Data["r_params_ins=$ins"]
    X_train = Input_Data["X_train_ins=$ins"]
    Y_train = Input_Data["Y_train_ins=$ins"]
    Z_train = Input_Data["Z_train_ins=$ins"]
    X_test = Input_Data["X_test_ins=$ins"]
    Y_test = Input_Data["Y_test_ins=$ins"]
    Z_test = Input_Data["Z_test_ins=$ins"]
    r0 = r_params.r0;
    r = r_params.r;
    alp0_true = theta_true.alpha0;
    alp_true = theta_true.alpha;
    beta_true = theta_true.beta;
    A_true = theta_true.A;

    lambda_len = length(lambda_list)
    gamma_len = length(gamma_list)

    for l_index in 1:lambda_len
        lambda = lambda_list[l_index]
        alpha0_hat = Estimate_Dict["alpha0_ins=$(ins)_lambda=$lambda"]
        alpha_hat = Estimate_Dict["alpha_ins=$(ins)_lambda=$lambda"]
        beta_hat = Estimate_Dict["beta_ins=$(ins)_lambda=$lambda"]
        A_hat = Estimate_Dict["A_ins=$(ins)_lambda=$lambda"]

        for g_index in 1:gamma_len
            gamma = gamma_list[g_index]

            obj_RO_list = zeros(n_test);
            x_RO_list = zeros(n_test,d);
            time_RO_list = zeros(n_test);
            profit_RO_list = zeros(n_test);
            for i in 1:n_test
                z_input = Z_test[i,:];
                obj_RO_list[i], x_RO_list[i,:], time_RO_list[i] = Robust_Product_Design(2, d, p, gamma, psi_lb, psi_ub, phi_lb, phi_ub, alpha0_hat, alpha_hat, beta_hat, A_hat, r0, r, z_input);
                profit_RO_list[i] = calculate_profit(alp0_true, alp_true, beta_true, A_true, r0, r, x_RO_list[i,:], z_input)
                # if i % 20 == 1
                #     println("RO: lambda=$lambda,gamma=$gamma, i=$i, obj=$(round(obj_RO_list[i], digits=4)), profit=$(round(profit_RO_list[i], digits=4)),x=$(round.(x_RO_list[i,:], digits=4)), time=$(round(time_RO_list[i],digits=4))")
                # end
            end
            Result_RO_Dict["obj_ins=$(ins)_lambda=$(lambda)_gamma=$gamma"] = obj_RO_list
            Result_RO_Dict["sol_ins=$(ins)_lambda=$(lambda)_gamma=$gamma"] = x_RO_list
            Result_RO_Dict["time_ins=$(ins)_lambda=$(lambda)_gamma=$gamma"] = time_RO_list
            Result_RO_Dict["profit_ins=$(ins)_lambda=$(lambda)_gamma=$gamma"] = profit_RO_list
        end
    end
    println("************ data_dir_ins: ", data_dir_ins,"*************")
end
save(string(data_dir,"Result_RO.jld2"),Result_RO_Dict) 