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


function Estimate_MNL_Para_Ridge(PM_train_extend, P_train, choice_train,S, N,lambda)

    U_hat = [PM_train_extend[s,:] for s in 1:S];
    P_hat = [P_train[s,:] for s in 1:S];
    choices = choice_train;

    # 初始化参数
    theta_init = randn(N * (2*N + 1));

    # 绑定目标函数和梯度
    od = Optim.OnceDifferentiable(
        θ -> neg_log_likelihood(θ, U_hat, P_hat, choices, N) + lambda * norm(θ, 1),
        (g, θ) -> neg_log_likelihood_grad!(g, θ, U_hat, P_hat, choices, N),
        theta_init
    )
    # 用 BFGS 优化
    result = optimize(od, theta_init, BFGS())
    theta_hat = Optim.minimizer(result)
    A_hat,B_hat = unpack_params(theta_hat, N);

    return A_hat,B_hat
end



function Estimate_OPT_Model(N,N_u,S,PM_train,P_train,choice_train,is_ridge,lbd)
    model = Model(Mosek.Optimizer)
    #（可选：设置求解器参数）
    set_attribute(model, "QUIET", true)
    # 变量定义
    @variable(model, B_esti[1:N,1:N])            
    @variable(model, A_esti[1:N,1:N_u])            
    @variable(model, w_0[1:N])            
    @variable(model, g[1:S])
    @variable(model, ell[1:S])          
    @variable(model, y0[1:S])          
    @variable(model, Y[1:S,1:N])                 

    for n in 1:N
        for j in 1:N
            if j != n
                @constraint(model, A_esti[n,j] == 0.0)
            end
        end    
    end

    for s in 1:S
        ind_s = Int(choice_train[s])
        if ind_s == N+1
            @constraint(model, 1 == g[s])
        else
            @constraint(model, PM_train[s,:]' * A_esti[ind_s,:] + P_train[s,:]' * B_esti[ind_s,:] + w_0[ind_s] == g[s])
        end
    end

    for s in 1:S
        @constraint(model, 1 >= y0[s] + sum(Y[s,1:N]))
    end

    for s in 1:S
        @constraint(model, [ -ell[s], 1.0, y0[s]] in MOI.ExponentialCone())
    end

    for s in 1:S
        for j in 1:N
            @constraint(model, [ PM_train[s,:]' * A_esti[j,:] + P_train[s,:]' * B_esti[j,:] + w_0[j] - ell[s], 1.0, Y[s,j]] in MOI.ExponentialCone())
        end
    end

    if is_ridge
        @variable(model, t)
        @constraint(model, [t;vec(B_esti);vec(A_esti);vec(w_0)] in MOI.NormOneCone(N*N + N*N_u + N + 1))
    end

    # 目标： maximize a_n^T w - v
    if is_ridge
        @objective(model, Max, (1/S) * (sum(g) - sum(ell)) - lbd * t)
    else
        @objective(model, Max, (1/S) * (sum(g) - sum(ell)))
    end
    optimize!(model)

    # 读取并打印结果
    status = termination_status(model)

    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        sol_status = string(status)
        obj_val = objective_value(model)
        A_hat = value.(A_esti)
        B_hat = value.(B_esti)
        Intercept = value.(w_0)
        solve_time = JuMP.solve_time(model)
    else
        sol_status = "Others"
        obj_val = NaN
        A_hat = ones(N,N_u) .* NaN
        B_hat = ones(N,N) .* NaN
        Intercept = ones(N) .* NaN
        solve_time = NaN
    end
    return A_hat,B_hat,Intercept,obj_val,sol_status,solve_time

end