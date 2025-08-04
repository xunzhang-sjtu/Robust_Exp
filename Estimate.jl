using Optim

# 将参数向量 vec_theta 展平为 a 和 b 的矩阵
function unpack_params(vec_theta, N, d_u, d_p)
    offset_a = N * d_u
    a = reshape(vec_theta[1:offset_a], (d_u, N))
    b = reshape(vec_theta[offset_a+1:end], (d_p, N))
    return a, b
end

# 构建负对数似然函数
function neg_log_likelihood(vec_theta::Vector{Float64},U_train, P_train, choices, N, d_u, d_p)
    a, b = unpack_params(vec_theta, N, d_u, d_p)

    total_log_likelihood = 0.0
    S = size(U_train, 1)  # 样本数量
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

        total_log_likelihood += log(prob + 1e-12)  # 加epsilon避免log(0)
    end

    return -total_log_likelihood
end
