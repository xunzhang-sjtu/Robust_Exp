function calculate_profit(params, r0, r, X_val, z_input)
    alp0 = params.alpha0
    alp = params.alpha
    beta = params.beta
    A = params.A

    V0 = 1.0
    utilities = zeros(N)
    for j in 1:N
        x_ij = X_val[j, :] # 第 j 个产品的设计
        # 计算效用 Uz(x_ij) = α₀* + <α*, x_ij> + <β*, z_i> + x_ij^T * A* * z_i
        utility = alp0 + dot(alp, x_ij) + dot(beta, z_input) + dot(x_ij, A * z_input) # x_ij^T * A* * z_i
        utilities[j] = utility
    end

    exp_utilities = exp.(utilities)
    # 计算分母 (V0 + sum(exp(Uz(x_il))))
    denominator = V0 + sum(exp_utilities)

    # 计算选择每个产品 j 的概率
    prob_choose_product = exp_utilities ./ denominator
    # 计算选择默认选项 (索引 0) 的概率
    prob_choose_default = V0 / denominator

    profits = r0 .+ X_val * r 
    total_profit = profits' * prob_choose_product
    return total_profit
end