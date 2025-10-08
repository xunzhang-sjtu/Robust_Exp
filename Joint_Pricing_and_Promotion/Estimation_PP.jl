# 将参数向量 vec_theta 展平为 a 和 b 的矩阵
function unpack_params(vec_theta, N)
    offset_a = N * (N+1)
    a = reshape(vec_theta[1:offset_a], (N, N+1))
    b = reshape(vec_theta[offset_a+1:end], (N, N))
    return a, b
end

# 构建负对数似然函数
function neg_log_likelihood(vec_theta::Vector{Float64}, U_hat, P_hat, choices, N)
    a, b = unpack_params(vec_theta, N)
    S = size(U_hat, 1)
    total_log_likelihood = 0.0
    for s in 1:S
        u = U_hat[s]
        p = P_hat[s]
        y = Int(choices[s])  # true label in {1,...,N,N+1}

        logits = [dot(a[n,:], u) + dot(b[n,:], p) for n in 1:N]

        exp_logits = exp.(logits)
        denom = 1 + sum(exp_logits)

        if y == (N+1)
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
                                  U_hat, P_hat, choices, N)
    a, b = unpack_params(vec_theta, N)
    S = size(U_hat, 1)

    grad_a = zeros(N,N+1)
    grad_b = zeros(N,N)

    for s in 1:S
        u = U_hat[s]
        p = P_hat[s]
        y = Int(choices[s])

        logits = [dot(a[n,:], u) + dot(b[n,:], p) for n in 1:N]
        exp_logits = exp.(logits)
        denom = 1 + sum(exp_logits)
        probs = exp_logits ./ denom
        baseline_prob = 1 / denom

        for n in 1:N
            indicator = (y == n) ? 1.0 : 0.0
            grad_a[n,:] .+= (indicator - probs[n]) .* u
            grad_b[n,:] .+= (indicator - probs[n]) .* p
        end
    end
    g .= -vcat(vec(grad_a), vec(grad_b))  # 负号因为是负对数似然
end

function Estimate_MNL_Para(PM_train_extend, P_train, choice_train,S, N)

    U_hat = [PM_train_extend[s,:] for s in 1:S];
    P_hat = [P_train[s,:] for s in 1:S];
    choices = choice_train;

    # 初始化参数
    theta_init = randn(N * (2*N + 1));

    # 绑定目标函数和梯度
    od = Optim.OnceDifferentiable(
        θ -> neg_log_likelihood(θ, U_hat, P_hat, choices, N),
        (g, θ) -> neg_log_likelihood_grad!(g, θ, U_hat, P_hat, choices, N),
        theta_init
    )
    # 用 BFGS 优化
    result = optimize(od, theta_init, BFGS())
    theta_hat = Optim.minimizer(result)
    A_hat,B_hat = unpack_params(theta_hat, N);

    return A_hat,B_hat
end

