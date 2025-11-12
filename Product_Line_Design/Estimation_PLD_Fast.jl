# ------------------------------------------------------------
# 1. 预计算所有扩展设计矩阵
# ------------------------------------------------------------
function precompute_extended_designs(X::Vector{Matrix{Float64}}, Z::Matrix{Float64})
    n = length(X)
    d = size(X[1], 2)
    p = size(Z, 2)
    total_dim = (d + 1) * (p + 1)
    X_tilde = Vector{Matrix{Float64}}(undef, n)

    # 预分配临时向量以避免重复分配
    x_tilde_temp = Vector{Float64}(undef, total_dim)

    for i in 1:n
        mi = size(X[i], 1)
        Xt = Matrix{Float64}(undef, mi, total_dim)

        @inbounds for j in 1:mi
            x_ij = @view X[i][j, :]
            z_i = @view Z[i, :]

            idx = 1
            x_tilde_temp[idx] = 1.0
            idx += 1

            @views x_tilde_temp[idx:idx+d-1] .= x_ij
            idx += d

            @inbounds for k in 1:p
                zk = z_i[k]
                x_tilde_temp[idx] = zk
                idx += 1
                @views x_tilde_temp[idx:idx+d-1] .= zk .* x_ij
                idx += d
            end

            @views Xt[j, :] .= x_tilde_temp
        end
        X_tilde[i] = Xt
    end
    return X_tilde
end

# ------------------------------------------------------------
# 2. 快速单样本负对数似然（使用预计算的 Xt_i）
# ------------------------------------------------------------
function neg_log_likelihood_single_fast(theta::AbstractVector, Xt_i::AbstractMatrix, y_i::Int)
    mi = size(Xt_i, 1)
    # 计算所有产品的效用：<θ, x̃_ij> for j=1..mi
    utilities = Xt_i * theta          # mi 维向量
    # 添加默认选项（效用为 0）
    m_val = maximum(utilities)
    # 数值稳定的 log-sum-exp: log(sum(exp(u))) = m + log(sum(exp(u - m)))
    shifted_utils = utilities .- m_val
    sum_exp = sum(exp, shifted_utils) + exp(-m_val)  # + exp(0 - m_val) for default option
    log_denominator = m_val + log(sum_exp)

    if y_i == 0
        log_prob = -log_denominator  # 因为 U0 = 0 → log(exp(0)/denom) = -log_denom
    else
        log_prob = utilities[y_i] - log_denominator
    end

    return -log_prob
end

# ------------------------------------------------------------
# 3. 快速整体负对数似然
# ------------------------------------------------------------
function neg_log_likelihood_fast(theta::AbstractVector, X_tilde::Vector{Matrix{Float64}}, Y::Vector{Int})
    n = length(Y)
    total_neg_ll = 0.0
    @inbounds for i in 1:n
        total_neg_ll += neg_log_likelihood_single_fast(theta, X_tilde[i], Y[i])
    end
    return total_neg_ll / n
end

# ------------------------------------------------------------
# 4. Lasso 目标函数（使用预计算数据）
# ------------------------------------------------------------
function lasso_objective_fast(theta::AbstractVector, lambda::Float64, X_tilde::Vector{Matrix{Float64}}, Y::Vector{Int})
    nll = neg_log_likelihood_fast(theta, X_tilde, Y)
    l1_penalty = lambda * norm(theta, 1)
    return nll + l1_penalty
end

# ------------------------------------------------------------
# 5. 参数解析（不变）
# ------------------------------------------------------------
function parameter_divide(theta_hat, d::Int, p::Int)
    alpha0_hat = 0.0
    alpha_hat = zeros(d)
    beta_hat = zeros(p)
    A_hat = zeros(d, p)

    idx = 1
    alpha0_hat = theta_hat[idx]
    idx += 1

    alpha_hat = theta_hat[idx:idx+d-1]
    idx += d

    for k in 1:p
        beta_hat[k] = theta_hat[idx]
        idx += 1
        A_hat[:, k] = theta_hat[idx:idx+d-1]
        idx += d
    end
    return alpha0_hat, alpha_hat, beta_hat, A_hat
end

# ------------------------------------------------------------
# 6. 主估计函数（优化版）
# ------------------------------------------------------------
function estimate_parameters_fast(
    X::Vector{Matrix{Float64}},
    Y::Vector{Int64},
    Z::Matrix{Float64},
    lambda::Float64,
    d::Int,
    p::Int;
    initial_theta = nothing,
    precomputed_X_tilde = nothing
)
    # 预计算扩展设计（如果未提供）
    if isnothing(precomputed_X_tilde)
        @time X_tilde = precompute_extended_designs(X, Z)
    else
        X_tilde = precomputed_X_tilde
    end

    total_dim = (d + 1) * (p + 1)
    if isnothing(initial_theta)
        initial_theta = randn(total_dim) * 0.1
    end

    # 目标函数闭包
    obj(theta) = lasso_objective_fast(theta, lambda, X_tilde, Y)

    # 使用 OWLQN 优化（专为 L1 设计）
    # result = optimize(
    #     obj,
    #     initial_theta,
    #     OWLQN(lambda),
    #     Optim.Options(show_trace = false, g_tol = 1e-6, iterations = 1000, time_limit = 300)
    # )
    result = optimize(obj, initial_theta, LBFGS(), Optim.Options(show_trace=false, g_tol=1e-6, iterations=1000,time_limit = 300))

    theta_hat = Optim.minimizer(result)
    alpha0_hat, alpha_hat, beta_hat, A_hat = parameter_divide(theta_hat, d, p)

    return alpha0_hat, alpha_hat, beta_hat, A_hat, result, X_tilde
end

# function Estimation_This(lambda,X_train,Y_train,Z_train,N_x,N_u)
#     alpha0_hat, alpha_hat, beta_hat, A_hat, opt_result, X_tilde = estimate_parameters_fast(X_train, Y_train, Z_train, lambda, N_x, N_u);
#     EST_Para = (alpha0=alpha0_hat, alpha=alpha_hat, beta=beta_hat, A=A_hat)
#     return EST_Para
# end