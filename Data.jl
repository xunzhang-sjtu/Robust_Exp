using LinearAlgebra
using Distributions
using Optim
using Random
using StatsFuns
using DataFrames
using StatsBase
# 设置随机种子
# Random.seed!(12)


function generate_strictly_row_diagonally_dominant(n::Int, max_offdiag,offdiag_sign)
    Mat = zeros(n, n)
    for i in 1:n
        # 在非对角线上生成随机数 [-max_offdiag, max_offdiag]
        off_diag = rand(n) .* (2max_offdiag) .- max_offdiag
        off_diag[i] = 0  # 避免给自己赋值两次

        # 计算非对角元素绝对值之和
        sum_offdiag = sum(abs, off_diag)

        # 设置对角元，使其严格大于其他元素之和
        diag_value = sum_offdiag + rand() * max_offdiag + 1e-3  # 加小量保证严格性

        if offdiag_sign == "positive" 
            Mat[i, :] = abs.(off_diag)
        end
        if offdiag_sign == "negative" 
            Mat[i, :] = -abs.(off_diag)
        end
        if offdiag_sign == "mix" 
            Mat[i, :] = off_diag
        end
        if offdiag_sign == "zero" 
            Mat[i, :] = off_diag .* 0.0
        end        
        Mat[i, i] = -abs.(diag_value)
    end
    return Mat
end


function Generate_Coef(N_u, N,max_offdiag,offdiag_sign)
    # N_u: 特征 u 的维度
    # N: 选项数量
    # A_true = [rand(Uniform(-1, 1), N_u) for n in 1:N]
    # A_true = [round.(a; digits=2) for a in A_true]

    # B_Mat = generate_strictly_row_diagonally_dominant(N::Int, max_offdiag,offdiag_sign)
    # B_true = [round.(B_Mat[n,:]; digits=2) for n in 1:N]

    # # B_true = [rand(Uniform(-1, 1), N_p) for n in 1:N]
    # # B_true = [round.(b; digits=2) for b in B_true]

    A_true = rand(N,N_u);
    B_true = generate_strictly_row_diagonally_dominant(N, max_offdiag,offdiag_sign);


    return A_true, B_true
end

function Generate_Feat_Data(d_u, d_p, S)
    # Step 2: 生成样本 (u, p)，均匀分布
    # U_train = [round.(randn(d_u),digits=2) for s in 1:S];
    # P_train = [round.(randn(d_p),digits=2) for s in 1:S];

    # U_train = [round.(rand(d_u),digits=2) for s in 1:S];
    # P_train = [round.(rand(d_p),digits=2) for s in 1:S];

    U_train = round.(rand(S,d_u),digits=2);
    P_train = round.(rand(S,d_p),digits=2);
    return U_train, P_train
end

function Calculate_Prob(S,N, U_train, P_train, a_star, b_star)
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

function Calculate_Choice(S,N,probs)
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

