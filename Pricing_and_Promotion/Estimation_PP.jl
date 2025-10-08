function neg_loglike(params, P_sample, PM_sample, choice_sample)
    N = size(P_sample, 2)
    a = params[1:N]
    b = params[N+1:2*N]
    c = params[2*N+1:3*N]
    S = size(P_sample, 1)

    nll = 0.0
    for s in 1:S
        price = P_sample[s, :]
        promo = PM_sample[s, :]
        util = exp.(a .+ b .* price .+ c .* promo)
        denom = 1 + sum(util)
        prob = vcat(util ./ denom, 1/denom) # N+1 choices
        chosen = Int(choice_sample[s])
        nll -= log(prob[chosen])
    end
    return nll
end

function Estimate(N,P_sample, PM_sample, choice_sample)
    params0 = randn(N,N)  # 初始值
    result = optimize(p -> neg_loglike(p, P_sample, PM_sample, choice_sample),
                    params0, LBFGS())

    params_est = Optim.minimizer(result)

    a_hat = params_est[1:N]
    b_hat = params_est[N+1:2*N]
    c_hat = params_est[2*N+1:3*N]
    return a_hat,b_hat,c_hat
end