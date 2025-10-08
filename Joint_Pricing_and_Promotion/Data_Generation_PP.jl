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

function Generate_Coef(N_u, N,max_offdiag,offdiag_sign)
    A_true = rand(N,N_u*N+1);
    for i in 0:(N_u-1)
        A_P = generate_strictly_row_diagonally_dominant(N, max_offdiag,"zero");
        A_true[1:N,(i*N+1):(i*N+N)] = A_P
    end
    B_true = generate_strictly_row_diagonally_dominant(N, max_offdiag,offdiag_sign);
    
    return A_true, B_true
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

# function Calculate_Prob(S,N, U_train, P_train, a_star, b_star)
#     # Step 3: 计算每个样本的选择概率
#     probs = Matrix{Float64}(undef, S, N)  # 每行一个样本，每列一个选项
#     for s in 1:S
#         u_this = U_train[s]
#         p_this = P_train[s]
#         logits = [dot(a_star[n], u_this) + dot(b_star[n], p_this) for n in 1:N]
#         exp_logits = exp.(logits)
#         denom = 1 + sum(exp_logits)
#         probs[s, :] = exp_logits ./ denom
#     end
#     return probs
# end

# function Calculate_Choice(S,N,probs)
#     # Step 4（可选）: 基于概率采样选项（含 baseline，编号为 0）
#     choices = Vector{Int}(undef, S)
#     for s in 1:S
#         p = probs[s, :]
#         baseline_prob = 1 - sum(p)
#         full_p = vcat(baseline_prob, p)  # 添加 baseline 的概率
#         choices[s] = sample(0:N, Weights(full_p))  # 随机选择（含baseline）
#     end
#     # # 打印部分结果
#     # df = DataFrame(choice = choices)
#     # println(first(df, 10))
#     return choices
# end

# function search_opt_price(N,p_lb,p_ub,b_n)
#     model = Model(Mosek.Optimizer)
#     set_attribute(model, "QUIET", true)
#     # 定义变量
#     @variable(model, price[1:N])                      # y_{nk}
#     @constraint(model, price .>= p_lb)
#     @constraint(model, price .<= p_ub)
#     @objective(model, Max,b_n' * price )
#     optimize!(model)
#     obj_val = objective_value(model)
#     return obj_val
# end

function generate_Input_Data(S_train,iterations, N, N_u, K, offdiag_sign,max_offdiag,P_bar)
    Input_Data = Dict()
    for iter in 1:iterations
        A_true, B_true = Generate_Coef(N_u, N, max_offdiag, offdiag_sign);
        P_train,PM_train,choice_train,PM_train_extend = Generate_Data(N,S_train,A_true,B_true,P_bar);

        Input_Data["iter=$(iter)_A_true"] = A_true;
        Input_Data["iter=$(iter)_B_true"] = B_true;
        Input_Data["iter=$(iter)_P_dag"] = round.(rand(N, K) .* P_bar; digits=2);
        Input_Data["iter=$(iter)_P_train"] = P_train;
        Input_Data["iter=$(iter)_PM_train_extend"] = PM_train_extend;
        Input_Data["iter=$(iter)_choice_train"] = choice_train;

        A_hat,B_hat = Estimate_MNL_Para(PM_train_extend, P_train, choice_train,S_train, N);

        Input_Data["iter=$(iter)_A_hat"] = A_hat
        Input_Data["iter=$(iter)_B_hat"] = B_hat
        println("****** iter = $(iter) ********")
    end
    return Input_Data
end

function Calculate_Hyper_Param(RO_coef_all, iterations, N, N_u, K, Input_Data)
    for iter in 1:iterations
        Obs_Feat = Input_Data["iter=$(iter)_Obs_Feat"]
        # A_true = Input_Data["iter=$(iter)_A_true"]
        # B_true = Input_Data["iter=$(iter)_B_true"]
        P_dag = Input_Data["iter=$(iter)_P_dag"]
        A_hat = Input_Data["iter=$(iter)_A_hat"]
        B_hat = Input_Data["iter=$(iter)_B_hat"]

        for RO_coef in RO_coef_all
            A_lb = A_hat .- RO_coef .* abs.(A_hat);
            A_ub = A_hat .+ RO_coef .* abs.(A_hat);
            B_lb = B_hat .- RO_coef .* abs.(B_hat);
            B_ub = B_hat .+ RO_coef .* abs.(B_hat);
            
            p_ub = vec(maximum(P_dag,dims=2))
            p_lb = vec(minimum(P_dag,dims=2))
            p_max = maximum(p_ub)
            p_min = minimum(p_lb)
            Obs_Feat_Trun = [max(-Obs_Feat[ind],0) for ind in 1:N_u]
            psi_lb = zeros(N)
            for n in 1:N
                b_n = B_lb[n,:]
                obj_n = search_opt_price(N,p_lb,p_ub,b_n)
                psi_lb[n] = max(-1000,-exp(-Obs_Feat_Trun' * (A_ub[n,:] - A_lb[n,:]) + Obs_Feat' * A_lb[n,:] + obj_n)*(p_max-p_min))
            end        

            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_psi_lb"] = psi_lb
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_psi_ub"] = zeros(N)
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_phi_lb"] = [p_min - p_max for i in 1:N]
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_phi_ub"] = zeros(N)

            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_A_lb"] = A_lb
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_A_ub"] = A_ub
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_B_lb"] = B_lb
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_B_ub"] = B_ub
        end
    end
    return Input_Data
end