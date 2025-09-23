function calculate_profit(alp0, alp, beta, A, r0, r, x_val, z_input)
    utility = alp0 + alp' * x_val + beta' * z_input + x_val' * A * z_input
    prob = exp(utility)/(1+exp(utility))
    profits = r0 + r' * x_val
    total_profit = profits * prob
    return total_profit
end