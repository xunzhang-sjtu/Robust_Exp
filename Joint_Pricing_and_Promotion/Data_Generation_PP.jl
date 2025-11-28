function generate_strictly_row_diagonally_dominant(n::Int, max_offdiag,offdiag_sign)
    Mat = zeros(n, n)
    for i in 1:n
        # 在非对角线上生成随机数 [-max_offdiag, max_offdiag]
        off_diag = rand(n) .* (2max_offdiag) .- max_offdiag
        off_diag[i] = 0  # 避免给自己赋值两次

        # 计算非对角元素绝对值之和
        sum_offdiag = sum(abs, off_diag)

        # 设置对角元，使其严格大于其他元素之和
        diag_value = sum_offdiag + rand() * max_offdiag + 1e-3  # 加小量保证严格性

        if offdiag_sign == "positive" 
            Mat[i, :] = abs.(off_diag)
        end
        if offdiag_sign == "negative" 
            Mat[i, :] = -abs.(off_diag)
        end
        if offdiag_sign == "mix" 
            Mat[i, :] = off_diag
        end
        if offdiag_sign == "zero" 
            Mat[i, :] = off_diag .* 0.0
        end        
        Mat[i, i] = -abs.(diag_value)
    end
    return Mat
end

function Generate_Coef(N_u, N,max_offdiag,offdiag_sign,is_original_setting)
    if is_original_setting
        a = [1.0,1.0,1.0]
        b = [-0.1,-0.2,-0.3]
        c = [0.8,0.3,0.5]
        B_true = generate_strictly_row_diagonally_dominant(N, max_offdiag,"zero");
        A_true = rand(N,N_u*N+1);
        for i in 0:(N_u-1)
            A_P = generate_strictly_row_diagonally_dominant(N, max_offdiag,"zero");
            A_true[1:N,(i*N+1):(i*N+N)] = A_P
        end
        A_true[:,N+1] = a
        for n in 1:N
            A_true[n,n] = c[n]
            B_true[n,n] = b[n]
        end
        w_true = ones(N*(N+N_u+1))
        n_w = N+N_u+1
        for j in 1:N
            w_true[(j-1) * n_w + 1] = A_true[j,N+1]
            w_true[(j-1) * n_w + 2] = A_true[j,j]
            w_true[((j-1) * n_w + 3):((j-1) * n_w +2+N)] = B_true[j,:]
        end
        w_true = [1.0,0.8,-0.1,1.0,0.3,-0.2,1.0,0.5,-0.3]
    else
        A_true = rand(N,N_u*N+1);
        for i in 0:(N_u-1)
            A_P = -generate_strictly_row_diagonally_dominant(N, max_offdiag,"zero");
            A_true[1:N,(i*N+1):(i*N+N)] = A_P
        end
        B_true = generate_strictly_row_diagonally_dominant(N, max_offdiag,offdiag_sign);
    end
    return A_true, B_true,w_true
end

function Generate_Data(N,S,A,B,P_bar)
    P_sample = round.(rand(S,N) .* P_bar,digits=2);

    indices = rand(1:N, S)
    PM_sample = zeros(Int, S, N)
    PM_sample[CartesianIndex.(1:S, indices)] .= 1
    PM_sample_extend = zeros(S,N+1)
    choice_sample = zeros(S)
    for s in 1:S
        price = P_sample[s,:]
        PP = vcat(PM_sample[s,:],1)
        PM_sample_extend[s,:] = PP
        utilities = ones(N+1) # the N+1 is the no purchase choice
        utilities[1:N] = exp.(vec(A * PP .+ B * price))
        prob = utilities ./sum(utilities)
        choice_sample[s] = sample(1:(N+1), Weights(prob))
    end
    return P_sample,PM_sample,choice_sample,PM_sample_extend
end



function Generate_Coef_Wide_Format(N_u, N,max_offdiag,offdiag_sign,is_original_setting)
    if is_original_setting
        w_true = [1.0,0.8,-0.1,1.0,0.3,-0.2,1.0,0.5,-0.3]
    else
        println("Has not Implemented generating random coefficients...")
    end
    return w_true
end

function Generate_Data_Wide_Format(N,N_u,S,w,P_bar)
    # P_sample = round.(rand(S,N) .* P_bar,digits=2);
    P_sample = 10 .+ round.(rand(S,N) .* 5,digits=2)

    indices = rand(1:N, S)
    PM_sample = zeros(Int, S, N)
    PM_sample[CartesianIndex.(1:S, indices)] .= 1
    choice_sample = zeros(S)
    Feature_sample = Vector{Matrix{Float64}}(undef, S)

    n_w = N
    for s in 1:S
        price_this = P_sample[s, :]
        prom_this = PM_sample[s, :]
        feature_this = zeros(N,N*N)
        for j in 1:N 
            feature_this[j, (j-1) * n_w + 1] = 1
            feature_this[j, (j-1) * n_w + 2] = prom_this[j]
            feature_this[j, (j-1) * n_w + 3] = price_this[j]
        end
        Feature_sample[s] = feature_this
    end

    V0 = 1.0
    for s in 1:S
        feature_this = Feature_sample[s]
        utilities = ones(N) # the N+1 is the no purchase choice
        for j in 1:N 
            utilities[j] = feature_this[j,:]' * w
        end
        exp_utilities = exp.(utilities)
        denominator = V0 + sum(exp_utilities)
        prob_choose_product = exp_utilities ./ denominator
        prob_choose_default = V0 / denominator
        choice_probs = vcat(prob_choose_default, prob_choose_product)
        choice = sample(0:N, Weights(choice_probs))
        choice_sample[s] = choice
    end
    return choice_sample,Feature_sample
end


# function generate_Input_Data(S_train,iterations, N, N_u, K, offdiag_sign,max_offdiag,P_bar,is_original_setting)
#     Input_Data = Dict()
#     for iter in 1:iterations
#         A_true, B_true = Generate_Coef(N_u, N, max_offdiag, offdiag_sign,is_original_setting);
#         P_train,PM_train,choice_train,PM_train_extend = Generate_Data(N,S_train,A_true,B_true,P_bar);

#         Input_Data["iter=$(iter)_A_true"] = A_true;
#         Input_Data["iter=$(iter)_B_true"] = B_true;
#         Input_Data["iter=$(iter)_P_dag"] = round.(rand(N, K) .* P_bar; digits=2);
#         Input_Data["iter=$(iter)_P_train"] = P_train;
#         Input_Data["iter=$(iter)_PM_train_extend"] = PM_train_extend;
#         Input_Data["iter=$(iter)_choice_train"] = choice_train;

#         A_hat,B_hat = Estimate_MNL_Para(PM_train_extend, P_train, choice_train,S_train, N);

#         Input_Data["iter=$(iter)_A_hat"] = A_hat
#         Input_Data["iter=$(iter)_B_hat"] = B_hat
#         println("****** iter = $(iter) ********")
#     end
#     return Input_Data
# end

# function Calculate_Hyper_Param(RO_coef_all, iterations, N, N_u, K, Input_Data)
#     for iter in 1:iterations
#         Obs_Feat = Input_Data["iter=$(iter)_Obs_Feat"]
#         # A_true = Input_Data["iter=$(iter)_A_true"]
#         # B_true = Input_Data["iter=$(iter)_B_true"]
#         P_dag = Input_Data["iter=$(iter)_P_dag"]
#         A_hat = Input_Data["iter=$(iter)_A_hat"]
#         B_hat = Input_Data["iter=$(iter)_B_hat"]

#         for RO_coef in RO_coef_all
#             A_lb = A_hat .- RO_coef .* abs.(A_hat);
#             A_ub = A_hat .+ RO_coef .* abs.(A_hat);
#             B_lb = B_hat .- RO_coef .* abs.(B_hat);
#             B_ub = B_hat .+ RO_coef .* abs.(B_hat);
            
#             p_ub = vec(maximum(P_dag,dims=2))
#             p_lb = vec(minimum(P_dag,dims=2))
#             p_max = maximum(p_ub)
#             p_min = minimum(p_lb)
#             Obs_Feat_Trun = [max(-Obs_Feat[ind],0) for ind in 1:N_u]
#             psi_lb = zeros(N)
#             for n in 1:N
#                 b_n = B_lb[n,:]
#                 obj_n = search_opt_price(N,p_lb,p_ub,b_n)
#                 psi_lb[n] = max(-1000,-exp(-Obs_Feat_Trun' * (A_ub[n,:] - A_lb[n,:]) + Obs_Feat' * A_lb[n,:] + obj_n)*(p_max-p_min))
#             end        

#             Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_psi_lb"] = psi_lb
#             Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_psi_ub"] = zeros(N)
#             Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_phi_lb"] = [p_min - p_max for i in 1:N]
#             Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_phi_ub"] = zeros(N)

#             Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_A_lb"] = A_lb
#             Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_A_ub"] = A_ub
#             Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_B_lb"] = B_lb
#             Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_B_ub"] = B_ub
#         end
#     end
#     return Input_Data
# end