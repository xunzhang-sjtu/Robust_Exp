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


function obtain_profits(RST_True_All, RST_ETO_All, RST_RO_All, instances, gamma_list)
    profit_True = zeros(instances);
    profit_ETO = zeros(instances);
    profit_RO = zeros(instances, length(gamma_list));
    for ins in 1:instances
        profit_True[ins] = mean(RST_True_All["RST_True_ins=$(ins)"]["profit"])
        profit_ETO[ins] = mean(RST_ETO_All["RST_ETO_ins=$(ins)"]["profit"])
        for g_index in 1:length(gamma_list)
            gamma = gamma_list[g_index]
            # println("gamma = $gamma")
            RST_RO_Gamma = RST_RO_All["RST_RO_ins=$(ins)"]["gamma=$gamma"]
            profit_RO[ins,g_index] = mean(RST_RO_Gamma["profit"])
        end
    end
    return profit_True, profit_ETO, profit_RO
end