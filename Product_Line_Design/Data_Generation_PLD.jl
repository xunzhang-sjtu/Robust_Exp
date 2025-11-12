function Generate_Wang_Qi_Max_True_Paras(d,p,s,coef_Params)
    # --- 步骤 1: 生成真实参数 θ* ---
    # alpha0_star = rand(Uniform(0.01, 0.02))
    # alpha_star = rand(Uniform(-1.0, 0.0), d)
    # beta_star = rand(Uniform(-0.02, 0.02), p)
    alpha0_star = rand(Uniform(coef_Params.alp0_lb, coef_Params.alp0_ub))
    alpha_star = rand(Uniform(coef_Params.alp_lb, coef_Params.alp_ub), d)
    beta_star = rand(Uniform(coef_Params.beta_lb, coef_Params.beta_ub), p)

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
        # A_star[row, col] = rand(Uniform(-0.02, 0.02))
        A_star[row, col] = rand(Uniform(coef_Params.A_lb, coef_Params.A_ub))
    end
    theta_true = (alpha0=alpha0_star, alpha=alpha_star, beta=beta_star, A=A_star)

    # --- 步骤 2: 生成收益参数 r ---
    # r0 = rand(Uniform(0.0, 1.0))
    # r = rand(Uniform(0.0, 0.1), d)
    r0 = rand(Uniform(coef_Params.r0_lb, coef_Params.r0_ub))
    r = rand(Uniform(coef_Params.r_lb, coef_Params.r_ub), d)
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
            # X_i[j, :] = rand(Uniform(0.0, 1.0), d)
            X_i[j, :] = rand(0.0:1.0, d)
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

function Generate_Data_this(N_x,N_u,N_nonzero,S_train,S_test,m,coef_Params)
    theta_true, r_params = Generate_Wang_Qi_Max_True_Paras(N_x,N_u,N_nonzero,coef_Params);
    X_train,Y_train,Z_train = Generate_Wang_Qi_Max_True_Data(N_x, N_u, S_train, m,theta_true);
    X_test,Y_test,Z_test = Generate_Wang_Qi_Max_True_Data(N_x, N_u, S_test, m,theta_true);
    Input_Data = Dict()
    Input_Data["theta_true"] = theta_true
    Input_Data["r_params"] = r_params
    Input_Data["X_train"] = X_train
    Input_Data["Y_train"] = Y_train
    Input_Data["Z_train"] = Z_train
    Input_Data["X_test"] = X_test
    Input_Data["Y_test"] = Y_test
    Input_Data["Z_test"] = Z_test
    return Input_Data
end

function Generate_Data_this_Same_Para(N_Max,N_x,N_u,S_train,S_test,theta_true_Fixed, r_params_Fixed)
    theta_true = theta_true_Fixed
    r_params = r_params_Fixed
    X_train,Y_train,Z_train = Generate_Wang_Qi_Max_True_Data(N_x, N_u, S_train, N_Max,theta_true);
    X_test,Y_test,Z_test = Generate_Wang_Qi_Max_True_Data(N_x, N_u, S_test, N_Max,theta_true);
    Input_Data = Dict()
    Input_Data["theta_true"] = theta_true
    Input_Data["r_params"] = r_params
    Input_Data["X_train"] = X_train
    Input_Data["Y_train"] = Y_train
    Input_Data["Z_train"] = Z_train
    Input_Data["X_test"] = X_test
    Input_Data["Y_test"] = Y_test
    Input_Data["Z_test"] = Z_test

    asorrtment_train = Array{Vector{Int64}}(undef,S_train)
    for s in 1:S_train
        asorrtment_train[s] = collect(1:N_Max)
    end
    Input_Data["asorrtment_train"] = asorrtment_train
    return Input_Data
end

function Get_Input_Data(Input_Data)
    theta_true = Input_Data["theta_true"]
    r_params = Input_Data["r_params"]
    X_train = Input_Data["X_train"]
    Y_train = Input_Data["Y_train"]
    Z_train = Input_Data["Z_train"]
    X_test = Input_Data["X_test"]
    Y_test = Input_Data["Y_test"]
    Z_test = Input_Data["Z_test"]
    asorrtment_train = Input_Data["asorrtment_train"]
    return theta_true,r_params,X_train,Y_train,Z_train,asorrtment_train,X_test,Y_test,Z_test
end