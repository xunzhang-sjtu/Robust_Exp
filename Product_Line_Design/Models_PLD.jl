function ETO_PLD(N,N_x,nu0,nu,r0,r,c_l,d_r,rev_gap,num_c,Time_Limit)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", Time_Limit) 
    # 定义变量
    @variable(model, rho_0 >= 0)                       
    @variable(model, rho[1:N] >= 0) 
    @variable(model, x[1:N,1:N_x], Bin)
    @variable(model, rho_0_X[1:N,1:N_x])
    @variable(model, rho_n_X[1:N,1:N_x])

    @constraint(model, rho_0 + sum(rho) == 1)

    for ind in 1:N
        @constraint(model, [nu0 * rho_0 + nu' * rho_0_X[ind,:] ,rho_0,rho[ind]] in MOI.ExponentialCone())
    end
    for ind1 in 1:N
        for ind2 in 1:N_x
            @constraint(model, rho_0_X[ind1,ind2] <= rho_0)
            @constraint(model, rho_0_X[ind1,ind2] <= x[ind1,ind2])
            @constraint(model, rho_0_X[ind1,ind2] >= x[ind1,ind2] - (1 - rho_0))
            @constraint(model, rho_0_X[ind1,ind2] >= 0)
        end
    end

    for ind in 1:N
        @constraint(model, [-nu0 * rho[ind] - nu' * rho_n_X[ind,:],rho[ind],rho_0] in MOI.ExponentialCone())
    end
    for ind1 in 1:N
        for ind2 in 1:N_x
            @constraint(model, rho_n_X[ind1,ind2] <= rho[ind1])
            @constraint(model, rho_n_X[ind1,ind2] <= x[ind1,ind2])
            @constraint(model, rho_n_X[ind1,ind2] >= x[ind1,ind2] - (1 - rho[ind1]))
            @constraint(model, rho_n_X[ind1,ind2] >= 0)
        end
    end


    # v_l = zeros(N_x)
    # start_index = 1
    # while start_index <= N_x
    #     end_index = start_index + num_c - 1
    #     end_index = min(end_index,N_x)
    #     v_l = zeros(N_x)
    #     v_l[start_index:end_index] = ones(end_index - start_index + 1)
    #     @constraint(model, x * v_l .== 1)
    #     start_index = start_index + num_c
    # end

    @constraint(model, x * c_l .>= d_r)
    for ind1 in 1:(N-1)
        @constraint(model, r' * x[ind1,:] >= r' * x[(ind1+1),:] + rev_gap)
    end

    @objective(model, Max,r0 * sum(rho) + sum(rho_n_X * r))

    optimize!(model)
    status = JuMP.termination_status(model)
    if status == MOI.OPTIMAL || status == MOI.TIME_LIMIT
        sol_status = string(status)
        obj_val = objective_value(model)
        X_val = value.(x)
        solve_time = JuMP.solve_time(model)
    else
        sol_status = "Others"
        obj_val = NaN
        X_val = ones(N_x) .* NaN
        solve_time = NaN
    end

    return obj_val, X_val,solve_time,sol_status
end


function RO_PLD(N,N_x,nu0,nu,r0,r,c_l,d_r,rev_gap,psi_lb,psi_ub,phi_lb,phi_ub,gamma,dual_norm,num_c,Time_Limit)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", Time_Limit) 
    # 定义变量
    @variable(model, delta)          
    @variable(model, eta0 >= 0)              
    @variable(model, lbd0)            
    @variable(model, xi0[1:N_x])   
    @variable(model, eta[1:N] >= 0) 
    @variable(model, lbd[1:N])            
    @variable(model, xi[1:N,1:N_x]) 

    # exponential variables
    @variable(model, psi_1[1:N])                   
    @variable(model, psi_2[1:N])                   
    @variable(model, psi_3[1:N])        
    @variable(model, phi_1[1:N])                   
    @variable(model, phi_2[1:N])                   
    @variable(model, phi_3[1:N])         

    @variable(model, X[1:N, 1:N_x], Bin)        # 二进制变量 x_{jk}
    @variable(model, Y[1:N,1:N_x])    
    @variable(model, Z[1:N,1:N_x])    

    @constraint(model, lbd0 + sum(psi_3) .== 0)

    @constraint(model, xi0 .+ sum(Y,dims=1)[1,:] .== 0)
    for ind1 in 1:N
        for ind2 in 1:N_x 
            @constraint(model, Y[ind1,ind2] >= psi_lb[ind1] * X[ind1,ind2])
            @constraint(model, Y[ind1,ind2] <= psi_ub[ind1] * X[ind1,ind2])
            @constraint(model, Y[ind1,ind2] >= psi_3[ind1] - psi_ub[ind1] * (1 - X[ind1,ind2]))
            @constraint(model, Y[ind1,ind2] <= psi_3[ind1] - psi_lb[ind1] * (1 - X[ind1,ind2]))
        end
    end

    for n in 1:N
        @constraint(model, lbd[n] - phi_3[n] == 0)
    end
    for n in 1:N
        @constraint(model, xi[n,:] .- Z[n,:] .== 0)
    end
    for ind1 in 1:N
        for ind2 in 1:N_x 
            @constraint(model, Z[ind1,ind2] >= phi_lb[ind1] * X[ind1,ind2])
            @constraint(model, Z[ind1,ind2] <= phi_ub[ind1] * X[ind1,ind2])
            @constraint(model, Z[ind1,ind2] >= phi_3[ind1] - phi_ub[ind1] * (1 - X[ind1,ind2]))
            @constraint(model, Z[ind1,ind2] <= phi_3[ind1] - phi_lb[ind1] * (1 - X[ind1,ind2]))
        end
    end

    @constraint(model, delta + sum(psi_2) + sum(phi_1) + eta0 * gamma - lbd0 * nu0 - nu' * xi0 == 0)

    for n in 1:N
        @constraint(model, delta + psi_1[n] + phi_2[n] + eta[n] * gamma - lbd[n] * nu0 - nu' * xi[n,:] - r0 - r' * X[n,:] == 0)
    end
    
    for n in 1:N
        @constraint(model, [psi_3[n], psi_2[n], psi_1[n]] in MOI.DualExponentialCone())
    end

    for n in 1:N
        @constraint(model, [phi_3[n],phi_2[n],phi_1[n]] in MOI.DualExponentialCone())
    end

    if dual_norm == 2
        @constraint(model, [eta0;lbd0;xi0] in SecondOrderCone())
        for n in 1:N
            @constraint(model, [eta[n];lbd[n];xi[n,:]] in SecondOrderCone())
        end
    end
    if dual_norm == 1
        @constraint(model, [eta0;lbd0;xi0] in MOI.NormOneCone(N_x+2))
        for n in 1:N
            @constraint(model, [eta[n];lbd[n];xi[n,:]] in MOI.NormOneCone(N_x+2))
        end
    end

    # v_l = zeros(N_x)
    # start_index = 1
    # while start_index <= N_x
    #     end_index = start_index + num_c - 1
    #     end_index = min(end_index,N_x)
    #     v_l = zeros(N_x)
    #     v_l[start_index:end_index] = ones(end_index - start_index + 1)
    #     @constraint(model, X * v_l .== 1)
    #     start_index = start_index + num_c
    # end

    @constraint(model, X * c_l .>= d_r)
    for ind1 in 1:N
        @constraint(model, c_l' * Y[ind1,:] <= d_r[ind1] * psi_3[ind1])
        @constraint(model, c_l' * Z[ind1,:] <= d_r[ind1] * phi_3[ind1])
    end

    for ind1 in 1:(N-1)
        @constraint(model, r' * X[ind1,:] >= r' * X[(ind1+1),:] + rev_gap)
    end

    @objective(model, Max, delta)
    
    optimize!(model)
    status = JuMP.termination_status(model)
    # println("status: ", status)
    # solution_summary(model)
    if status == MOI.OPTIMAL  || status == MOI.TIME_LIMIT
        sol_status = string(status)
        obj_val = objective_value(model)
        X_val = round.(value.(X))
        solve_time = JuMP.solve_time(model)
    else
        sol_status = "Others"
        obj_val = NaN
        X_val = ones(N,N) .* NaN
        solve_time = NaN
    end
    return obj_val, X_val,solve_time,sol_status
end