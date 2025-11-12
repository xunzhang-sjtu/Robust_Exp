
function Estimate_OPT_Model_PLD(N_Max,N_x,N_u,choice_train,X_train,Z_train, asorrtment_train,is_ridge, lbd)
    """
    Parameters:
    - N_x: dimension of product features
    - N_u: dimension of customer features
    - S: training data size
    - N_Max: maximum assortment size
    - utility = alpha0 + alpha^T X + beta^T z + X^T A z
    - choice_train: training choice data, a vector of length S
    - X_train: training product features, a vector of S elements, each is a matrix of size (N, N_x)
    - Z_train: training customer features, a matrix of size (S, N_u)
    - assortment_train: training assortment data, a vector of S elements, each is a vector of assortment indices
    - is_ridge: whether to use ridge regularization
    - lbd: regularization coefficient for ridge (if is_ridge is true)
    """
    S = length(choice_train)
    model = Model(Mosek.Optimizer)
    #（set paramters for MOSEK）
    set_attribute(model, "QUIET", true)
    @variable(model, alpha[1:N_x])            
    @variable(model, beta[1:N_u])            
    @variable(model, alpha0)          
    @variable(model, A[1:N_x,1:N_u])   
    @variable(model, g[1:S])
    @variable(model, ell[1:S])          
    @variable(model, y0[1:S])          
    @variable(model, Y[1:S,1:N_Max])         

    all_assortment = collect(1:N_Max)
    for s in 1:S
        ind_s = Int(choice_train[s])
        if ind_s == 0
            @constraint(model, 1 == g[s])
        else
            @constraint(model, alpha0 + X_train[s,:][1][ind_s,:]' * alpha + Z_train[s,:]' * beta + X_train[s,:][1][ind_s,:]' * A * Z_train[s,:] == g[s])
        end
    end

    for s in 1:S
        @constraint(model, 1 >= y0[s] + sum(Y[s,1:N_Max]))
    end

    for s in 1:S
        @constraint(model, [ -ell[s], 1.0, y0[s]] in MOI.ExponentialCone())
    end

    for s in 1:S
        assortment = asorrtment_train[s]
        setdiff_assortment = setdiff(all_assortment, assortment)
        for j in assortment
            @constraint(model, [ alpha0 + X_train[s,:][1][j,:]' * alpha + Z_train[s,:]' * beta + X_train[s,:][1][j,:]' * A * Z_train[s,:] - ell[s], 1.0, Y[s,j]] in MOI.ExponentialCone())
        end
        for j in setdiff_assortment
            @constraint(model, Y[s,j] == 0)
        end
    end

    # whether to use ridge regularization
    if is_ridge
        @variable(model, t)
        @constraint(model, [t;alpha0;alpha;beta;vec(A)] in MOI.NormOneCone(2+ N_x + N_u + N_x * N_u))
    end

    # Set objective
    if is_ridge
        @objective(model, Max, (1/S) * (sum(g) - sum(ell)) - lbd * t)
    else
        @objective(model, Max, (1/S) * (sum(g) - sum(ell)))
    end
    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        sol_status = string(status)
        obj_val = objective_value(model)
        alp_0_exp = value.(alpha0)
        alpha_exp = value.(alpha)
        beta_exp = value.(beta)
        A_exp = value.(A)
        solve_time = JuMP.solve_time(model)
    else
        sol_status = "Others"
        obj_val = NaN
        alp_0_exp = NaN
        alpha_exp = ones(N_x) .* NaN
        beta_exp = ones(N_u) .* NaN
        A_exp = ones(N_x,N_u) .* NaN
        solve_time = NaN
    end
    return alp_0_exp, alpha_exp, beta_exp, A_exp, solve_time, sol_status, obj_val
end


# function create_extended_design(x_ij::AbstractVector, z_i::AbstractVector)
#     """
#     根据论文 2.2 节，将原始数据转换为扩展设计向量 x̃ij。
#     """
#     d = length(x_ij)
#     p = length(z_i)

#     x_tilde = Vector{Float64}(undef, (d+1)*(p+1))
#     idx = 1

#     # 第一部分: 1 和 x_ij
#     x_tilde[idx] = 1.0
#     idx += 1
#     x_tilde[idx:idx+d-1] = x_ij
#     idx += d

#     # 第二部分: 交互项 (z_ik 和 z_ik * x_ij)
#     for k in 1:p
#         x_tilde[idx] = z_i[k] # z_ik
#         idx += 1
#         # z_ik * x_ij
#         x_tilde[idx:idx+d-1] = z_i[k] * x_ij
#         idx += d
#     end

#     return x_tilde
# end

# function neg_log_likelihood_single(theta::AbstractVector, X_i::AbstractMatrix, z_i::AbstractVector, y_i::Int)
#     """
#     计算单个样本 i 的负对数似然。

#     参数:
#     - θ: 参数向量，维度为 (d+1)*(p+1)
#     - X_i: 一个 mi x d 矩阵，代表第 i 个样本中的 mi 个产品设计。
#     - z_i: 一个 p 维向量，代表第 i 个样本中顾客的特征。
#     - y_i: 一个整数 (0, 1, ..., mi)，代表第 i 个样本中顾客的选择。0 表示默认选项。

#     返回:
#     - 单个样本的负对数似然值。
#     """
#     mi, d = size(X_i)
#     p = length(z_i)

#     # 计算所有备选项（包括默认选项）的效用指数 (未归一化的 logit)
#     utilities = zeros(mi + 1) # 索引 1 对应产品1, ..., 索引 mi 对应产品mi, 索引 mi+1 对应默认选项
#     for j in 1:mi
#         x_ij = X_i[j, :]
#         x_tilde_ij = create_extended_design(x_ij, z_i)
#         utilities[j] = dot(theta, x_tilde_ij) # <θ, x̃ij>
#     end
#     utilities[mi+1] = 0.0 # 默认选项的效用指数设为 0 (因为 V0 = exp(U0) = 1)

#     # 计算 log-sum-exp 以数值稳定的方式计算 log(分母)
#     log_denominator = log(sum(exp.(utilities)))

#     # 计算负对数似然
#     if y_i == 0
#         # 选择了默认选项 (索引为 mi+1)
#         neg_ll = - (utilities[mi+1] - log_denominator)
#     else
#         # 选择了第 y_i 个产品 (索引为 y_i)
#         neg_ll = - (utilities[y_i] - log_denominator)
#     end

#     return neg_ll
# end

# function neg_log_likelihood(theta::AbstractVector, X,Y,Z)
#     """
#     计算整个数据集 D 的负对数似然。

#     参数:
#     - θ: 参数向量，维度为 (d+1)*(p+1)
#     - D: 数据集，是一个元组向量 [(X_1, z_1, y_1), (X_2, z_2, y_2), ..., (X_n, z_n, y_n)]

#     返回:
#     - 整个数据集的平均负对数似然值。
#     """
#     n = length(X)
#     total_neg_ll = 0.0

#     for i in 1:n
#         total_neg_ll += neg_log_likelihood_single(theta, X[i], Z[i,:], Y[i])
#     end

#     return total_neg_ll / n
# end

# function lasso_objective(theta::AbstractVector, lambda::Float64,X,Y,Z)
#     """
#     定义 Lasso-正则化的 MLE 目标函数。

#     参数:
#     - θ: 参数向量，维度为 (d+1)*(p+1)
#     - D: 数据集，是一个元组向量 [(X_1, z_1, y_1), (X_2, z_2, y_2), ..., (X_n, z_n, y_n)]
#     - λ: L1 正则化系数 (λn)

#     返回:
#     - 目标函数值：平均负对数似然 + λ * ||θ||_1
#     """
#     nll = neg_log_likelihood(theta, X,Y,Z)
#     l1_penalty = lambda * norm(theta, 1) # L1 范数
#     return nll + l1_penalty
# end

# function parameter_divide(theta_hat,d,p)
#     alpha0_hat = 0.0
#     alpha_hat = zeros(d)
#     beta_hat = zeros(p)
#     A_hat = zeros(d,p)

#     idx = 1
#     # 第一部分: 1 和 x_ij
#     alpha0_hat = theta_hat[idx]
#     idx += 1

#     alpha_hat = theta_hat[idx:idx+d-1]
#     idx += d

#     # 第二部分: 交互项 (z_ik 和 z_ik * x_ij)
#     for k in 1:p
#         beta_hat[k] = theta_hat[idx] # z_ik
#         idx += 1
#         # z_ik * x_ij
#         A_hat[:,k] = theta_hat[idx:idx+d-1]
#         idx += d
#     end
#     return alpha0_hat, alpha_hat, beta_hat, A_hat
# end

# function estimate_parameters(X::Vector{Matrix{Float64}},Y::Vector{Int64},Z::Matrix{Float64},lambda::Float64, d::Int, p::Int; initial_theta=nothing)
#     """
#     执行 Lasso-正则化 MLE 估计。

#     参数:
#     - D: 数据集，是一个元组向量 [(X_1, z_1, y_1), (X_2, z_2, y_2), ..., (X_n, z_n, y_n)]
#     - λ: L1 正则化系数 (λn)
#     - d: 产品特征维度
#     - p: 顾客特征维度
#     - initial_θ: (可选) 初始参数值。如果未提供，则随机初始化。

#     返回:
#     - θ_hat: 估计的参数向量。
#     - result: Optim.jl 的优化结果对象，包含收敛信息等。
#     """

#     # 设置初始值
#     if isnothing(initial_theta)
#         initial_theta = randn((d+1)*(p+1)) * 0.1 # 小随机初始化
#     end
#     # # 定义目标函数 (只接受 θ 作为参数)
#     objective_function(theta) = lasso_objective(theta,lambda, X,Y,Z)
#     # 使用 L-BFGS 进行优化
#     # 注意: L1 正则项不可导，但 L-BFGS 通常能处理得很好，或者可以使用专门为 L1 设计的算法 (如 OWLQN)
#     result = optimize(objective_function, initial_theta, LBFGS(), Optim.Options(show_trace=false, g_tol=1e-6, iterations=1000))
    
#     # result = optimize(objective_function, initial_theta, OWLQN(lambda), Optim.Options(show_trace=false, g_tol=1e-6, iterations=1000))

#     theta_hat = Optim.minimizer(result)

#     alpha0_hat, alpha_hat, beta_hat, A_hat = parameter_divide(theta_hat,d,p)

#     return alpha0_hat, alpha_hat, beta_hat, A_hat, result
# end

