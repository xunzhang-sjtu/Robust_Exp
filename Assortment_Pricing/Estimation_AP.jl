function compute_prob(S,N, U_train, P_train, a_star, b_star)    
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

function generate_choice(S,N,probs)
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

# 将参数向量 vec_theta 展平为 a 和 b 的矩阵
function unpack_params(vec_theta, N, d_u, d_p)
    offset_a = N * d_u
    a = reshape(vec_theta[1:offset_a], (d_u, N))
    b = reshape(vec_theta[offset_a+1:end], (d_p, N))
    return a, b
end

# 构建负对数似然函数
function neg_log_likelihood(vec_theta::Vector{Float64}, U_train, P_train, choices, N, d_u, d_p)
    a, b = unpack_params(vec_theta, N, d_u, d_p)
    S = size(U_train, 1)

    total_log_likelihood = 0.0
    for s in 1:S
        u = U_train[s]
        p = P_train[s]
        y = choices[s]  # true label in {0,1,...,N}

        logits = [dot(a[:,n], u) + dot(b[:,n], p) for n in 1:N]
        exp_logits = exp.(logits)
        denom = 1 + sum(exp_logits)

        if y == 0
            prob = 1 / denom
        else
            prob = exp_logits[y] / denom
        end

        total_log_likelihood += log(prob + 1e-12)  # 加 epsilon 避免 log(0)
    end

    return -total_log_likelihood
end

# 梯度函数
function neg_log_likelihood_grad!(g::Vector{Float64}, vec_theta::Vector{Float64}, 
                                  U_train, P_train, choices, N, d_u, d_p)
    a, b = unpack_params(vec_theta, N, d_u, d_p)
    S = size(U_train, 1)

    grad_a = zeros(d_u, N)
    grad_b = zeros(d_p, N)

    for s in 1:S
        u = U_train[s]
        p = P_train[s]
        y = choices[s]

        logits = [dot(a[:,n], u) + dot(b[:,n], p) for n in 1:N]
        exp_logits = exp.(logits)
        denom = 1 + sum(exp_logits)
        probs = exp_logits ./ denom
        baseline_prob = 1 / denom

        for n in 1:N
            indicator = (y == n) ? 1.0 : 0.0
            grad_a[:, n] .+= (indicator - probs[n]) .* u
            grad_b[:, n] .+= (indicator - probs[n]) .* p
        end
    end

    g .= -vcat(vec(grad_a), vec(grad_b))  # 负号因为是负对数似然
end

function Estimate_MNL_Para(U_input, P_input, S, N, d_u, d_p, A_true, B_true)

    a_star = [A_true[n,:] for n in 1:N]
    b_star = [B_true[n,:] for n in 1:N]
    U_train = [U_input[s,:] for s in 1:S]
    P_train = [P_input[s,:] for s in 1:S]

    probs = compute_prob(S, N, U_train, P_train, a_star, b_star)
    choices = generate_choice(S, N, probs)

    # 初始化参数
    theta_init = randn(N * (d_u + d_p))

    # 绑定目标函数和梯度
    od = Optim.OnceDifferentiable(
        θ -> neg_log_likelihood(θ, U_train, P_train, choices, N, d_u, d_p),
        (g, θ) -> neg_log_likelihood_grad!(g, θ, U_train, P_train, choices, N, d_u, d_p),
        theta_init
    )

    # 用 BFGS 优化
    result = optimize(od, theta_init, BFGS())

    theta_hat = Optim.minimizer(result)
    a_hat, b_hat = unpack_params(theta_hat, N, d_u, d_p)

    # println("a_hat:")
    # println(Matrix(a_hat'))
    # println("b_hat:")
    # println(Matrix(b_hat'))
    return Matrix(a_hat'), Matrix(b_hat')
end

