# ************************ Assortment Pricing Functions *****************************
# - search_opt_price()
# - generate_Input_Data()
# Calculate_Hyper_Param()
# ************************************************************************************

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



# ************************ Functions in Wang_Qi_Max *****************************
# - Generate_Wang_Qi_Max_True_Paras()
# - Generate_Wang_Qi_Max_True_Data()
# ************************************************************************************

function Generate_Wang_Qi_Max_True_Paras(d,p,s)
    # --- 步骤 1: 生成真实参数 θ* ---
    alpha0_star = rand(Uniform(1.0, 2.0))
    alpha_star = rand(Uniform(-1.0, 1.0), d)
    beta_star = rand(Uniform(-2.0, 2.0), p)

    # 生成稀疏交互矩阵 A*
    A_star = zeros(d, p)
    # 随机选择 s 个不同的位置 (线性索引)
    total_elements = d * p
    # 确保 s 不超过总元素数
    s_actual = min(s, total_elements)
    # 无放回地随机选择 s_actual 个位置
    selected_indices = sample(1:total_elements, s_actual, replace=false)
    # 为选中的位置赋值
    for idx in selected_indices
        row, col = Tuple(CartesianIndices((d, p))[idx])
        A_star[row, col] = rand(Uniform(-2.0, 2.0))
    end

    theta_true = (alpha0=alpha0_star, alpha=alpha_star, beta=beta_star, A=A_star)

    # --- 步骤 2: 生成收益参数 r ---
    r0 = rand(Uniform(0.0, 1.0))
    r = rand(Uniform(-1.0, 1.0), d)
    r_params = (r0=r0, r=r)

    return theta_true, r_params

end

function Generate_Wang_Qi_Max_True_Data(d, p, n, m,theta_true)

    # 初始化存储
    X = Vector{Matrix{Float64}}(undef, n); # n 个 m×d 矩阵
    Z = Matrix{Float64}(undef, n, p);      # n×p 矩阵
    Y = Vector{Int}(undef, n);             # n 维向量

    for i in 1:n
        z_i = rand(Uniform(-1.0, 1.0), p)
        Z[i, :] = z_i

        X_i = zeros(m, d)
        for j in 1:m
            X_i[j, :] = rand(Uniform(0.0, 1.0), d)
        end
        X[i] = X_i

        # --- 计算选择概率 ---
        # 根据公式 (2.1): Pz(x; θ*) = exp(Uz(x)) / (V0 + exp(Uz(x)))
        # 论文中 V0 = exp(U0) = 1 (默认选项效用权重归一化为1)
        V0 = 1.0
        utilities = zeros(m)
        for j in 1:m
            x_ij = X_i[j, :] # 第 j 个产品的设计
            # 计算效用 Uz(x_ij) = α₀* + <α*, x_ij> + <β*, z_i> + x_ij^T * A* * z_i
            utility = theta_true.alpha0 +
                        dot(theta_true.alpha, x_ij) +
                        dot(theta_true.beta, z_i) +
                        dot(x_ij, theta_true.A * z_i) # x_ij^T * A* * z_i
            utilities[j] = utility
        end

        # 计算分子 exp(Uz(x_ij))
        exp_utilities = exp.(utilities)
        # 计算分母 (V0 + sum(exp(Uz(x_il))))
        denominator = V0 + sum(exp_utilities)

        # 计算选择每个产品 j 的概率
        prob_choose_product = exp_utilities ./ denominator
        # 计算选择默认选项 (索引 0) 的概率
        prob_choose_default = V0 / denominator

        # 构建完整的概率向量 [P(选择默认), P(选择产品1), ..., P(选择产品m_actual)]
        choice_probs = vcat(prob_choose_default, prob_choose_product)
        # println("Choice probabilities: ", choice_probs)
        # --- 进行多项抽样得到选择结果 y_i ---
        # 使用 StatsBase 的 sample 函数
        # 选择结果: 0 表示默认选项, 1 表示第一个产品, ..., m_actual 表示第 m_actual 个产品
        y_i = sample(0:m, Weights(choice_probs))
        Y[i] = y_i
    end
    return X,Y,Z
end