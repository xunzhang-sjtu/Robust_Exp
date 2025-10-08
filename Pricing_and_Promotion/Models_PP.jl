function ETO_PP(N,K,a,b,c,P_dag)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # model = Model(COPT.ConeOptimizer)
    # 定义变量
    @variable(model, rho_0 >= 0);                       # ρ₀ ≥ 0
    @variable(model, rho[1:N] >= 0);                    # ρ_n ≥ 0

    @variable(model, omega_P[1:N]);                     
    @variable(model, omega_X[1:N]);
    @variable(model, pi_P[1:N]);                     
    @variable(model, pi_X[1:N]);

    @variable(model, x[1:N], Bin);                      # promotion indicator
    @variable(model, Y[1:N, 1:K],Bin);                  # price indicator
    @variable(model, rho_0_x[1:N]);                     
    @variable(model, rho_0_Y[1:N, 1:K]);                
    @variable(model, rho_x[1:N]);                     
    @variable(model, rho_Y[1:N, 1:K]); 

    @constraint(model, rho_0 + sum(rho) == 1)

    for n in 1:N
        @constraint(model, [a[n] * rho_0 + b[n] * omega_P[n] + c[n] * omega_X[n],rho_0,rho[n]] in MOI.ExponentialCone())
    end
    @constraint(model, omega_P .== sum(rho_0_Y .* P_dag, dims=2)[:,1])
    for n in 1:N
        for k in 1:K
            @constraint(model, 0 <= rho_0_Y[n,k])
            @constraint(model, rho_0_Y[n,k] <= Y[n, k])
            @constraint(model, rho_0_Y[n,k] >= rho_0 - (1 - Y[n, k]))
            @constraint(model, rho_0_Y[n,k] <= rho_0)
        end
    end

    @constraint(model, omega_X .== rho_0_x)
    for n in 1:N
        @constraint(model, 0 <= rho_0_x[n])
        @constraint(model, rho_0_x[n] <= x[n])
        @constraint(model, rho_0_x[n] >= rho_0 - (1 - x[n]))
        @constraint(model, rho_0_x[n] <= rho_0)
    end

    for n in 1:N
        @constraint(model, [-a[n] * rho[n] - b[n] * pi_P[n] - c[n] * pi_X[n], rho[n], rho_0] in MOI.ExponentialCone())
    end
    @constraint(model, pi_P .== sum(rho_Y .* P_dag, dims=2)[:,1])
    for n in 1:N
        for k in 1:K
            @constraint(model, 0 <= rho_Y[n,k])
            @constraint(model, rho_Y[n,k] <= Y[n, k])
            @constraint(model, rho_Y[n,k] >= rho[n] - (1 - Y[n, k]))
            @constraint(model, rho_Y[n,k] <= rho[n])
        end
    end
    @constraint(model, pi_X .== rho_x)
    for n in 1:N
        @constraint(model, 0 <= rho_x[n])
        @constraint(model, rho_x[n] <= x[n])
        @constraint(model, rho_x[n] >= rho[n] - (1 - x[n]))
        @constraint(model, rho_x[n] <= rho[n])
    end

    @constraint(model, Y * ones(K) .== 1)
    @constraint(model, sum(x) == 1)
    @objective(model, Max,sum(rho_Y .* P_dag))
    optimize!(model)
    status = JuMP.termination_status(model)
    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        Y_val = value.(Y)
        x_val = value.(x)
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        Y_val = ones(N,K) .* NaN
        x_val = ones(N) .* NaN
        solve_time = NaN
    end
    return obj_val,Y_val,x_val,solve_time
end