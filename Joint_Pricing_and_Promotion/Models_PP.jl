function Solve_ETO(N,N_u,K,A,B,P_dag,Time_Limit)
    N_u = N
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", Time_Limit) 
    # model = Model(COPT.ConeOptimizer)
    # 定义变量
    @variable(model, rho_0 >= 0)                       # ρ₀ ≥ 0
    @variable(model, rho[1:N] >= 0)                    # ρ_n ≥ 0
    @variable(model, omega_0[1:(N_u+1)])               
    @variable(model, pi_0[1:N])                      
    @variable(model, omega[1:N,1:(N_u+1)])             
    @variable(model, PI[1:N,1:N])  
    @variable(model, X[1:N, 1:K], Bin)                 
    @variable(model, u[1:N_u], Bin)                 
    @variable(model, Y[1:N,1:N,1:K])                  
    @variable(model, Y_0[1:N, 1:K])                     

    @constraint(model, rho_0 + sum(rho) == 1)

    for n in 1:N
        @constraint(model, [A[n,:]' * omega_0 + B[n,:]' * pi_0,rho_0,rho[n]] in MOI.ExponentialCone())
    end
    @constraint(model, omega_0[N+1] == rho_0)
    for j in 1:N_u
        @constraint(model, 0 <= omega_0[j])
        @constraint(model, omega_0[j] <= u[j])
        @constraint(model, omega_0[j] >= rho_0 - (1 - u[j]))
        @constraint(model, omega_0[j] <= rho_0)
    end
    @constraint(model, pi_0 .== sum(Y_0 .* P_dag, dims=2))
    for n in 1:N
        for k in 1:K
            @constraint(model, 0 <= Y_0[n, k])
            @constraint(model, Y_0[n, k] <= X[n, k])
            @constraint(model, Y_0[n, k] >= rho_0 - (1 - X[n, k]))
            @constraint(model, Y_0[n, k] <= rho_0)
        end
    end

    for n in 1:N
        @constraint(model, [-A[n,:]' * omega[n,:] - B[n,:]' * PI[n,:], rho[n], rho_0] in MOI.ExponentialCone())
    end
    @constraint(model, omega[:,N_u + 1] .- rho .== 0)
    for n in 1:N
        for j in 1:N_u
            @constraint(model, omega[n,j] >= 0)
            @constraint(model, omega[n,j] <= u[j])
            @constraint(model, omega[n,j] >= rho[n] - (1-u[j]))
            @constraint(model, omega[n,j] <= rho[n])
        end
    end
    for n in 1:N
        @constraint(model, PI[n,:] .== sum(Y[n,:,:] .* P_dag, dims=2))
    end
    for n in 1:N
        for j in 1:N
            for k in 1:K
                # z_{nk} bounds
                @constraint(model, 0 <= Y[n,j, k])
                @constraint(model, Y[n,j,k] <= X[j, k])
                @constraint(model, Y[n,j,k] >= rho[n] - (1 - X[j, k]))
                @constraint(model, Y[n,j, k] <= rho[n])
            end
        end
    end

    @constraint(model, sum(X,dims=2) .== 1)
    @constraint(model, sum(u) .== 1)

    @objective(model, Max,sum([Y[n,n,:]' * P_dag[n,:] for n in 1:N]))
    optimize!(model)
    status = JuMP.termination_status(model)
    # solution_summary(model)
    # println("status = ",status)
    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        sol_status = string(status)
        obj_val = objective_value(model)
        X_val = value.(X)
        promo = value.(u)
        solve_time = JuMP.solve_time(model)
    else
        sol_status = "Others"
        obj_val = NaN
        X_val = ones(N,N) .* NaN
        promo = ones(N) .* NaN
        solve_time = NaN
    end
    # price = sum(X_val .* P_dag,dims=2)[:,1]
    # promo = value.(u)
    # println("obj = ",obj_val)
    # println("price = ",price)
    # println("promotion = ",promo)
    return obj_val, X_val,promo,solve_time,sol_status
end

function Solve_RO(N,N_u,K,A_hat,B_hat,P_dag,psi_lb,psi_ub,phi_lb,phi_ub,gamma,dual_norm,Time_Limit)
    N_u = N 
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", Time_Limit) 
    # 定义变量
    @variable(model, delta);                           # 标量 δ
    @variable(model, pi_0[1:N,1:(N_u+1)]);            
    @variable(model, theta_0[1:N,1:N]);           
    @variable(model, PI[1:N,1:(N_u+1)]);            
    @variable(model, Theta[1:N,1:N]);     
    @variable(model,lbd_0[1:N] >= 0);
    @variable(model,LBD[1:N] >= 0);

    # exponential variables
    @variable(model, psi_1[1:N]);                   
    @variable(model, psi_2[1:N]);                   
    @variable(model, psi_3[1:N]);        
    @variable(model, phi_1[1:N]);                   
    @variable(model, phi_2[1:N]);                   
    @variable(model, phi_3[1:N]);         

    @variable(model, X[1:N, 1:K], Bin);         # 二进制变量 x_{jk}
    @variable(model, u[1:N], Bin);              # 二进制变量 x_{jk}
    @variable(model, Y_0[1:N,1:N_u]);    
    @variable(model, Y[1:N,1:N,1:K]);    
    @variable(model, Z_0[1:N,1:N_u]);    
    @variable(model, Z[1:N,1:N,1:K]);

    @constraint(model, pi_0[:,N_u+1] .+ psi_3 .== 0)
    @constraint(model, pi_0[1:N,1:N_u] .+ Y_0 .== 0)
    for n in 1:N
        for j in 1:N_u
            @constraint(model, Y_0[n,j] >= psi_lb[n] * u[j])
            @constraint(model, Y_0[n,j] <= psi_ub[n] * u[j])
            @constraint(model, Y_0[n,j] >= psi_3[n] - psi_ub[n] * (1 - u[j]))
            @constraint(model, Y_0[n,j] <= psi_3[n] - psi_lb[n] * (1 - u[j]))
        end
    end
    for n in 1:N
        @constraint(model, theta_0[n,:] .+ sum(Y[n,:,:] .* P_dag,dims=2) .== 0)
    end
    for n in 1:N
        for j in 1:N
            for k in 1:K
                @constraint(model, Y[n,j,k] >= psi_lb[n] * X[j,k])
                @constraint(model, Y[n,j,k] <= psi_ub[n] * X[j,k])
                @constraint(model, Y[n,j,k] >= psi_3[n] - psi_ub[n] * (1 - X[j,k]))
                @constraint(model, Y[n,j,k] <= psi_3[n] - psi_lb[n] * (1 - X[j,k]))
            end
        end
    end

    @constraint(model, PI[:,N_u+1] .- phi_3 .== 0)
    @constraint(model, PI[1:N,1:N_u] .- Z_0 .== 0)
    for n in 1:N
        for j in 1:N
            @constraint(model, Z_0[n,j] >= phi_lb[n] * u[j])
            @constraint(model, Z_0[n,j] <= phi_ub[n] * u[j])
            @constraint(model, Z_0[n,j] >= phi_3[n] - phi_ub[n] * (1 - u[j]))
            @constraint(model, Z_0[n,j] <= phi_3[n] - phi_lb[n] * (1 - u[j]))
        end
    end
    for n in 1:N
        @constraint(model, Theta[n,:] .- sum(Z[n,:,:] .* P_dag,dims=2) .== 0)
    end
    for n in 1:N
        for j in 1:N
            for k in 1:K
                @constraint(model, Z[n,j,k] >= phi_lb[n] * X[j,k])
                @constraint(model, Z[n,j,k] <= phi_ub[n] * X[j,k])
                @constraint(model, Z[n,j,k] >= phi_3[n] - phi_ub[n] * (1 - X[j,k]))
                @constraint(model, Z[n,j,k] <= phi_3[n] - phi_lb[n] * (1 - X[j,k]))
            end
        end
    end

    @constraint(model, gamma' * lbd_0 - sum(A_hat .* pi_0) - sum(B_hat .* theta_0) + sum(psi_2) + sum(phi_1) + delta <= 0)
    for n in 1:N
        @constraint(model, LBD[n] * gamma[n] - A_hat[n,:]' * PI[n,:] - B_hat[n,:]' * Theta[n,:] + psi_1[n] + phi_2[n] + delta - X[n,:]' * P_dag[n,:] <= 0)
    end

    for n in 1:N
        @constraint(model, [psi_3[n], psi_2[n], psi_1[n]] in MOI.DualExponentialCone())
    end
    for n in 1:N
        @constraint(model, [phi_3[n],phi_2[n],phi_1[n]] in MOI.DualExponentialCone())
    end
    if dual_norm == 2
        for n in 1:N
            @constraint(model, [lbd_0[n];pi_0[n,:];theta_0[n,:]] in SecondOrderCone())
            @constraint(model, [LBD[n];PI[n,:];Theta[n,:]] in SecondOrderCone())
        end
    end
    if dual_norm == 1
        for n in 1:N
            @constraint(model, [lbd_0[n];pi_0[n,:];theta_0[n,:]] in MOI.NormOneCone(N + N_u + 2))
            @constraint(model, [LBD[n];PI[n,:];Theta[n,:]] in MOI.NormOneCone(N + N_u + 2))
        end
    end

    @constraint(model, sum(X,dims=2) .== 1)
    for n in 1:N
        @constraint(model, sum(Y_0[n,:]) == psi_3[n])
        @constraint(model, sum(Z_0[n,:]) == phi_3[n])
        for j in 1:N
            @constraint(model, sum(Y[n,j,:]) == psi_3[n])
            @constraint(model, sum(Z[n,j,:]) == phi_3[n])
        end
    end
    @constraint(model, sum(u) .== 1)
    @objective(model, Max, delta)

    optimize!(model)
    status = JuMP.termination_status(model)
    # solution_summary(model)
    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        sol_status = string(status)
        obj_val = objective_value(model)
        X_val = value.(X)
        promo = value.(u)
        solve_time = JuMP.solve_time(model)
    else
        sol_status = "Others"
        obj_val = NaN
        X_val = ones(N,N) .* NaN
        promo = ones(N) .* NaN
        solve_time = NaN
    end
    return obj_val, X_val,promo,solve_time,sol_status
end


function Solve_ETO_Compact(N,K,w,P_dag,Time_Limit)
    N_w = length(w)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", Time_Limit) 
    # 定义变量
    @variable(model, rho_0 >= 0)                       # ρ₀ ≥ 0
    @variable(model, rho[1:N] >= 0)                    # ρ_n ≥ 0
    @variable(model, PI_0[1:N,1:N_w])
    @variable(model, PI[1:N,1:N_w])                      
    @variable(model, P_Bin[1:N, 1:K], Bin)   # indicate price bin chosen
    @variable(model, Prom[1:N], Bin)         # indicate promotion chosen 
                
    @variable(model, Y_0[1:N])     # rho_0 * Prom[n]
    @variable(model, Z_0[1:N, 1:K]) # rho_0 * P_Bin[n,k]
    @variable(model, Y[1:N])     # rho[n] * Prom[n]
    @variable(model, Z[1:N, 1:K]) # rho[n] * P_Bin[n,k]

    @constraint(model, rho_0 + sum(rho) == 1)
    for n in 1:N
        @constraint(model, [w' * PI_0[n,:],rho_0,rho[n]] in MOI.ExponentialCone())
    end
    for n in 1:N 
        all_indices = collect(1:N*N)
        this_indices = collect((n-1)*N + 1:(n-1)*N + 3)
        diff_indices = setdiff(all_indices, this_indices)
        # println("n=$n,this_indices=",this_indices,",diff_indices=",diff_indices)
        for j in diff_indices
            @constraint(model, PI_0[n,j] == 0)
        end

        @constraint(model, PI_0[n,this_indices[1]] == rho_0)
        @constraint(model, PI_0[n,this_indices[2]] == Y_0[n])
        @constraint(model, PI_0[n,this_indices[3]] == Z_0[n,:]' * P_dag[n,:])
    end
    for n in 1:N
        @constraint(model, 0 <= Y_0[n])
        @constraint(model, Y_0[n] <= Prom[n])
        @constraint(model, Y_0[n] >= rho_0 - (1 - Prom[n]))
        @constraint(model, Y_0[n] <= rho_0)
    end
    for n in 1:N
        for k in 1:K
            @constraint(model, 0 <= Z_0[n, k])
            @constraint(model, Z_0[n, k] <= P_Bin[n, k])
            @constraint(model, Z_0[n, k] >= rho_0 - (1 - P_Bin[n, k]))
            @constraint(model, Z_0[n, k] <= rho_0)
        end
    end
    for n in 1:N
        @constraint(model, [-w' * PI[n,:], rho[n], rho_0] in MOI.ExponentialCone())
    end
    for n in 1:N 
        all_indices = collect(1:N*N)
        this_indices = collect((n-1)*N + 1:(n-1)*N + 3)
        diff_indices = setdiff(all_indices, this_indices)
        for j in diff_indices
            @constraint(model, PI[n,j] == 0)
        end

        @constraint(model, PI[n,this_indices[1]] == rho[n])
        @constraint(model, PI[n,this_indices[2]] == Y[n])
        @constraint(model, PI[n,this_indices[3]] == Z[n,:]' * P_dag[n,:])
    end
    for n in 1:N
        @constraint(model, 0 <= Y[n])
        @constraint(model, Y[n] <= Prom[n])
        @constraint(model, Y[n] >= rho[n] - (1 - Prom[n]))
        @constraint(model, Y[n] <= rho[n])
    end

    for n in 1:N
        for k in 1:K
            @constraint(model, 0 <= Z[n, k])
            @constraint(model, Z[n, k] <= P_Bin[n, k])
            @constraint(model, Z[n, k] >= rho[n] - (1 - P_Bin[n, k]))
            @constraint(model, Z[n, k] <= rho[n])
        end
    end
    @constraint(model, sum(P_Bin,dims=2) .== 1)
    @constraint(model, sum(Prom) == 1)

    @objective(model, Max,sum([Z[n,:]' * P_dag[n,:] for n in 1:N]))
    optimize!(model)
    status = JuMP.termination_status(model)
    # solution_summary(model)
    # println("status = ",status)
    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        sol_status = string(status)
        obj_val = objective_value(model)
        price_indices = value.(P_Bin)
        promo = value.(Prom)
        solve_time = JuMP.solve_time(model)
    else
        sol_status = "Others"
        obj_val = NaN
        price_indices = ones(N,K) .* NaN
        promo = ones(N) .* NaN
        solve_time = NaN
    end

    # price = [price_indices[n,:]' * P_dag[n,:] for n in 1:N]
    # println("Optimal objective value: ", obj_val)
    # println("Optimal prices: ", price)
    # println("Optimal promotion: ", promo)
    return obj_val, price_indices,promo,solve_time,sol_status
end

function Solve_RO_Compact(N,K,w,P_dag,psi_lb,psi_ub,phi_lb,phi_ub,gamma,dual_norm,Time_Limit)
    N_w = length(w)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", Time_Limit) 
    # 定义变量
    @variable(model, delta);                           
    # exponential variables
    @variable(model, psi_1[1:N]);                   
    @variable(model, psi_2[1:N]);                   
    @variable(model, psi_3[1:N]);        
    @variable(model, phi_1[1:N]);                   
    @variable(model, phi_2[1:N]);                   
    @variable(model, phi_3[1:N]);      

    @variable(model, Alpha[0:N,1:N_w]); 
    @variable(model, kappa[0:N]);            

    @variable(model, P_Bin[1:N, 1:K], Bin)  # indicate price choice        
    @variable(model, Prom[1:N], Bin)  # indicate promotion choice

    @variable(model, Y_0[1:N]);  # indicate psi_3[n] * Prom[n]
    @variable(model, Y[1:N]);    # indicate phi_3[n] * Prom[n]
    @variable(model, Z_0[1:N,1:K]);    # indicate psi_3[n] * P_Bin[n,k]
    @variable(model, Z[1:N,1:K]);   # indicate phi_3[n] * P_Bin[n,k]

    for n in 1:N 
        all_indices = collect(1:N*N)
        this_indices = collect((n-1)*N + 1:(n-1)*N + 3)
        diff_indices = setdiff(all_indices, this_indices)
        # println("n=$(n),this_indices=$(this_indices),diff_indices=$(diff_indices)")
        @constraint(model, Alpha[0,this_indices[1]] + psi_3[n] == 0)
        @constraint(model, Alpha[0,this_indices[2]] + Y_0[n] == 0)
        @constraint(model, Alpha[0,this_indices[3]] + P_dag[n,:]' * Z_0[n,:] == 0)
    end

    for n in 1:N
        @constraint(model, Y_0[n] >= psi_lb[n] * Prom[n])
        @constraint(model, Y_0[n] <= psi_ub[n] * Prom[n])
        @constraint(model, Y_0[n] >= psi_3[n] - psi_ub[n] * (1 - Prom[n]))
        @constraint(model, Y_0[n] <= psi_3[n] - psi_lb[n] * (1 - Prom[n]))
    end

    for n in 1:N
        for k in 1:K
            @constraint(model, Z_0[n,k] >= psi_lb[n] * P_Bin[n,k])
            @constraint(model, Z_0[n,k] <= psi_ub[n] * P_Bin[n,k])
            @constraint(model, Z_0[n,k] >= psi_3[n] - psi_ub[n] * (1 - P_Bin[n,k]))
            @constraint(model, Z_0[n,k] <= psi_3[n] - psi_lb[n] * (1 - P_Bin[n,k]))
        end
    end

    for n in 1:N 
        all_indices = collect(1:N*N)
        this_indices = collect((n-1)*N + 1:(n-1)*N + 3)
        diff_indices = setdiff(all_indices, this_indices)
        for j in diff_indices
            @constraint(model, Alpha[n,j] == 0)
        end
        @constraint(model, Alpha[n,this_indices[1]] - phi_3[n] == 0)
        @constraint(model, Alpha[n,this_indices[2]] - Y[n] == 0)
        @constraint(model, Alpha[n,this_indices[3]] - P_dag[n,:]' * Z[n,:] == 0)
    end

    for n in 1:N
        @constraint(model, Y[n] >= phi_lb[n] * Prom[n])
        @constraint(model, Y[n] <= phi_ub[n] * Prom[n])
        @constraint(model, Y[n] >= phi_3[n] - phi_ub[n] * (1 - Prom[n]))
        @constraint(model, Y[n] <= phi_3[n] - phi_lb[n] * (1 - Prom[n]))
    end

    for n in 1:N
        for k in 1:K
            @constraint(model, Z[n,k] >= phi_lb[n] * P_Bin[n,k])
            @constraint(model, Z[n,k] <= phi_ub[n] * P_Bin[n,k])
            @constraint(model, Z[n,k] >= phi_3[n] - phi_ub[n] * (1 - P_Bin[n,k]))
            @constraint(model, Z[n,k] <= phi_3[n] - phi_lb[n] * (1 - P_Bin[n,k]))
        end
    end

    @constraint(model, delta + gamma * kappa[0] - w' * Alpha[0,:] + sum(psi_2) + + sum(phi_1) == 0)
    for n in 1:N
        @constraint(model, delta + gamma * kappa[n] - w' * Alpha[n,:] + psi_1[n] + phi_2[n] == P_dag[n,:]' * P_Bin[n,:])
    end

    for n in 1:N
        @constraint(model, [psi_3[n], psi_2[n], psi_1[n]] in MOI.DualExponentialCone())
    end
    for n in 1:N
        @constraint(model, [phi_3[n],phi_2[n],phi_1[n]] in MOI.DualExponentialCone())
    end

    if dual_norm == 2
        for n in 0:N
            @constraint(model, [kappa[n];Alpha[n,:]] in SecondOrderCone())
        end
    end
    if dual_norm == 1
        for n in 0:N
            @constraint(model, [kappa[n];Alpha[n,:]] in MOI.NormOneCone(1+N_w))
        end
    end

    @constraint(model, sum(P_Bin,dims=2) .== 1)
    @constraint(model, sum(Prom) .== 1)

    @objective(model, Max, delta)

    optimize!(model)
    status = JuMP.termination_status(model)
    # solution_summary(model)
    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        sol_status = string(status)
        obj_val = objective_value(model)
        Price_indices = value.(P_Bin)
        promo = value.(Prom)
        solve_time = JuMP.solve_time(model)
    else
        sol_status = "Others"
        obj_val = NaN
        Price_indices = ones(N,K) .* NaN
        promo = ones(N) .* NaN
        solve_time = NaN
    end
    return obj_val, Price_indices,promo,solve_time,sol_status
end