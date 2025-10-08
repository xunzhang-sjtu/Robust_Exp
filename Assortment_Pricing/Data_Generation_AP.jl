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
    # N_u: 特征 u 的维度
    # N: 选项数量
    # A_true = [rand(Uniform(-1, 1), N_u) for n in 1:N]
    # A_true = [round.(a; digits=2) for a in A_true]

    # B_Mat = generate_strictly_row_diagonally_dominant(N::Int, max_offdiag,offdiag_sign)
    # B_true = [round.(B_Mat[n,:]; digits=2) for n in 1:N]

    # # B_true = [rand(Uniform(-1, 1), N_p) for n in 1:N]
    # # B_true = [round.(b; digits=2) for b in B_true]

    A_true = rand(N,N_u);
    B_true = generate_strictly_row_diagonally_dominant(N, max_offdiag,offdiag_sign);


    return A_true, B_true
end

function Generate_Feat_Data(d_u, d_p, S)
    # Step 2: 生成样本 (u, p)，均匀分布
    # U_train = [round.(randn(d_u),digits=2) for s in 1:S];
    # P_train = [round.(randn(d_p),digits=2) for s in 1:S];

    # U_train = [round.(rand(d_u),digits=2) for s in 1:S];
    # P_train = [round.(rand(d_p),digits=2) for s in 1:S];

    U_train = round.(rand(S,d_u),digits=2);
    P_train = round.(rand(S,d_p),digits=2);
    return U_train, P_train
end

function Calculate_Prob(S,N, U_train, P_train, a_star, b_star)
    # Step 3: 计算每个样本的选择概率
    probs = Matrix{Float64}(undef, S, N)  # 每行一个样本，每列一个选项
    for s in 1:S
        u_this = U_train[s]
        p_this = P_train[s]
        logits = [dot(a_star[n], u_this) + dot(b_star[n], p_this) for n in 1:N]
        exp_logits = exp.(logits)
        denom = 1 + sum(exp_logits)
        probs[s, :] = exp_logits ./ denom
    end
    return probs
end

function Calculate_Choice(S,N,probs)
    # Step 4（可选）: 基于概率采样选项（含 baseline，编号为 0）
    choices = Vector{Int}(undef, S)
    for s in 1:S
        p = probs[s, :]
        baseline_prob = 1 - sum(p)
        full_p = vcat(baseline_prob, p)  # 添加 baseline 的概率
        choices[s] = sample(0:N, Weights(full_p))  # 随机选择（含baseline）
    end
    # # 打印部分结果
    # df = DataFrame(choice = choices)
    # println(first(df, 10))
    return choices
end

function search_opt_price(N,p_lb,p_ub,b_n)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, price[1:N])                      # y_{nk}
    @constraint(model, price .>= p_lb)
    @constraint(model, price .<= p_ub)
    @objective(model, Max,b_n' * price )
    optimize!(model)
    obj_val = objective_value(model)
    return obj_val
end

function generate_Input_Data(S_train,S_test,iterations, N, N_u, K, offdiag_sign,max_offdiag,P_bar)
    Input_Data = Dict()
    for iter in 1:iterations
        A_true, B_true = Generate_Coef(N_u, N,max_offdiag,offdiag_sign);
        U_train, P_train = Generate_Feat_Data(N_u, N, S_train);
        U_test, P_test = Generate_Feat_Data(N_u, N, S_test);

        Input_Data["iter=$(iter)_Obs_Feat"] = U_test[1,:];
        Input_Data["iter=$(iter)_A_true"] = A_true;
        Input_Data["iter=$(iter)_B_true"] = B_true;
        Input_Data["iter=$(iter)_P_dag"] = round.(rand(N, K) .* P_bar; digits=2);
        Input_Data["iter=$(iter)_U_train"] = U_train;
        Input_Data["iter=$(iter)_P_train"] = P_train;

        A_hat, B_hat = Estimate_MNL_Para(U_train, P_train, S_train, N, N_u, N, A_true, B_true);

        Input_Data["iter=$(iter)_A_hat"] = A_hat
        Input_Data["iter=$(iter)_B_hat"] = B_hat
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