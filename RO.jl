using JuMP
using MosekTools

function Solve_RO(N,N_u,K,A_lb,A_ub,B_lb,B_ub,u,p_dag,psi_lb,psi_ub,phi_lb,phi_ub)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, delta)                           # 标量 δ
    @variable(model, omega_lb[1:N,1:N_u] >= 0)            
    @variable(model, omega_ub[1:N,1:N_u] >= 0)           
    @variable(model, pi_lb[1:N,1:N] >= 0)     
    @variable(model, pi_ub[1:N,1:N] >= 0)     
    @variable(model, eta_lb[1:N,1:N_u] >= 0)            
    @variable(model, eta_ub[1:N,1:N_u] >= 0)           
    @variable(model, ups_lb[1:N,1:N] >= 0)     
    @variable(model, ups_ub[1:N,1:N] >= 0) 
    # exponential variables
    @variable(model, psi_1[1:N])                   
    @variable(model, psi_2[1:N])                   
    @variable(model, psi_3[1:N])        
    @variable(model, phi_1[1:N])                   
    @variable(model, phi_2[1:N])                   
    @variable(model, phi_3[1:N])         

    @variable(model, X[1:N, 1:K], Bin)        # 二进制变量 x_{jk}
    @variable(model, Y[1:N,1:N,1:K])    
    @variable(model, Z[1:N,1:N,1:K])    

    for n in 1:N
        @constraint(model, omega_lb[n,:] .- omega_ub[n,:] .+ psi_3[n] .* u .== 0)
    end

    for n in 1:N
        @constraint(model, pi_lb[n,:] .- pi_ub[n,:] .+ sum(Y[n,:,:] .* p_dag,dims=2) .== 0)
    end

    for n in 1:N
        @constraint(model, eta_lb[n,:] .- eta_ub[n,:] .- phi_3[n] .* u .== 0)
    end

    for n in 1:N
        @constraint(model, ups_lb[n,:] .- ups_ub[n,:] .- sum(Z[n,:,:] .* p_dag,dims=2) .== 0)
    end

    @constraint(model, sum(omega_ub .* A_ub) - sum(omega_lb .* A_lb) + sum(pi_ub .* B_ub) - sum(pi_lb .* B_lb) + delta + sum(psi_2) + sum(phi_1) <= 0)

    for n in 1:N
        @constraint(model, eta_ub[n,:]' * A_ub[n,:] - eta_lb[n,:]' * A_lb[n,:] + ups_ub[n,:]' * B_ub[n,:] - ups_lb[n,:]' * B_lb[n,:] + delta + psi_1[n] + phi_2[n] - X[n,:]' * p_dag[n,:] <= 0)
    end

    for n in 1:N
        @constraint(model, [psi_3[n], psi_2[n], psi_1[n]] in MOI.DualExponentialCone())
    end

    for n in 1:N
        @constraint(model, [phi_3[n],phi_2[n],phi_1[n]] in MOI.ExponentialCone())
    end

    for n in 1:N
        for j in 1:N
            for k in 1:K
                @constraint(model, Y[n,j,k] >= psi_lb * X[j,k])
                @constraint(model, Y[n,j,k] <= psi_ub * X[j,k])
                # @constraint(model, Y[n,j,k] >= psi_3[n] - psi_ub * (1 - X[j,k]))
                # @constraint(model, Y[n,j,k] <= psi_3[n] - psi_lb * (1 - X[j,k]))

            end
        end
    end

    for n in 1:N
        for j in 1:N
            for k in 1:K
                @constraint(model, Z[n,j,k] >= phi_lb * X[j,k])
                @constraint(model, Z[n,j,k] <= phi_ub * X[j,k])
                # @constraint(model, Z[n,j,k] >= phi_3[n] - phi_ub * (1 - X[j,k]))
                # @constraint(model, Z[n,j,k] <= phi_3[n] - phi_lb * (1 - X[j,k]))
            end
        end
    end

    @constraint(model, sum(X,dims=2) .== 1)

    for n in 1:N
        for j in 1:N
            @constraint(model, sum(Y[n,j,:]) == psi_3[n])
            @constraint(model, sum(Z[n,j,:]) == phi_3[n])
        end
    end


    @objective(model, Max, delta)

    optimize!(model)
    status = JuMP.termination_status(model)
    # println("status: ", status)
    # solution_summary(model)
    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        X_val = round.(value.(X))
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        X_val = ones(N,N) .* NaN
        solve_time = NaN
    end
    return obj_val,X_val,solve_time
end

function Solve_RO_with_Price(N,N_u,A_lb,A_ub,B_lb,B_ub,u,price)
    model = Model(Mosek.Optimizer)
    # set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, delta)                           # 标量 δ
    @variable(model, omega_lb[1:N,1:N_u] >= 0)            
    @variable(model, omega_ub[1:N,1:N_u] >= 0)           
    @variable(model, pi_lb[1:N,1:N] >= 0)     
    @variable(model, pi_ub[1:N,1:N] >= 0)     
    @variable(model, eta_lb[1:N,1:N_u] >= 0)            
    @variable(model, eta_ub[1:N,1:N_u] >= 0)           
    @variable(model, ups_lb[1:N,1:N] >= 0)     
    @variable(model, ups_ub[1:N,1:N] >= 0) 
    # exponential variables
    @variable(model, psi_1[1:N])                   
    @variable(model, psi_2[1:N])                   
    @variable(model, psi_3[1:N])        
    @variable(model, phi_1[1:N])                   
    @variable(model, phi_2[1:N])                   
    @variable(model, phi_3[1:N])         

    for n in 1:N
        @constraint(model, omega_lb[n,:] .- omega_ub[n,:] .+ psi_3[n] .* u .== 0)
    end

    for n in 1:N
        @constraint(model, pi_lb[n,:] .- pi_ub[n,:] .+ psi_3[n] .* price .== 0)
    end

    for n in 1:N
        @constraint(model, eta_lb[n,:] .- eta_ub[n,:] .- phi_3[n] .* u .== 0)
    end

    for n in 1:N
        @constraint(model, ups_lb[n,:] .- ups_ub[n,:] .- phi_3[n] .* price .== 0)
    end

    @constraint(model, sum(omega_ub .* A_ub) - sum(omega_lb .* A_lb) + sum(pi_ub .* B_ub) - sum(pi_lb .* B_lb) + delta + sum(psi_2) + sum(phi_1) <= 0)

    for n in 1:N
        @constraint(model, eta_ub[n,:]' * A_ub[n,:] - eta_lb[n,:]' * A_lb[n,:] + ups_ub[n,:]' * B_ub[n,:] - ups_lb[n,:]' * B_lb[n,:] + delta + psi_1[n] + phi_2[n] - price[n] <= 0)
    end

    for n in 1:N
        @constraint(model, [psi_3[n], psi_2[n], psi_1[n]] in MOI.DualExponentialCone())
    end

    for n in 1:N
        @constraint(model, [phi_3[n],phi_2[n],phi_1[n]] in MOI.ExponentialCone())
    end

    @objective(model, Max, delta)

    optimize!(model)
    status = JuMP.termination_status(model)
    println("status: ", status)
    solution_summary(model)
    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        solve_time = NaN
    end
    return obj_val,solve_time
end


function Solve_RO_one_side_exp(N,N_u,K,A_lb,A_ub,B_lb,B_ub,u,p_dag,psi_lb,psi_ub)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, delta)                           # 标量 δ
    @variable(model, omega_lb[1:N,1:N_u] >= 0)            
    @variable(model, omega_ub[1:N,1:N_u] >= 0)           
    @variable(model, pi_lb[1:N,1:N] >= 0)     
    @variable(model, pi_ub[1:N,1:N] >= 0)     

    # exponential variables
    @variable(model, psi_1[1:N])                   
    @variable(model, psi_2[1:N])                   
    @variable(model, psi_3[1:N])        
    
    @variable(model, X[1:N, 1:K], Bin)        # 二进制变量 x_{jk}
    @variable(model, Y[1:N,1:N,1:K] <= 0)    

    for n in 1:N
        @constraint(model, omega_lb[n,:] .- omega_ub[n,:] .+ psi_3[n] .* u .== 0)
    end

    for n in 1:N
        @constraint(model, pi_lb[n,:] .- pi_ub[n,:] .+ sum(Y[n,:,:] .* p_dag,dims=2) .== 0)
    end

    @constraint(model, sum(omega_ub .* A_ub) - sum(omega_lb .* A_lb) + sum(pi_ub .* B_ub) - sum(pi_lb .* B_lb) + delta + sum(psi_2) <= 0)

    for n in 1:N
        @constraint(model, delta + psi_1[n] - X[n,:]' * p_dag[n,:] <= 0)
    end

    for n in 1:N
        @constraint(model, [psi_3[n], psi_2[n], psi_1[n]] in MOI.DualExponentialCone())
    end


    # for n in 1:N
    #     for j in 1:N
    #         for k in 1:K
    #             @constraint(model, Y[n,j,k] >= psi_lb * X[j,k])
    #             @constraint(model, Y[n,j,k] <= psi_ub * X[j,k])
    #             @constraint(model, Y[n,j,k] >= psi_3[n] - psi_ub * (1 - X[j,k]))
    #             @constraint(model, Y[n,j,k] <= psi_3[n] - psi_lb * (1 - X[j,k]))
    #         end
    #     end
    # end

    for n in 1:N
        for j in 1:N
            for k in 1:K
                @constraint(model, Y[n,j,k] >= psi_lb[n] * X[j,k])
                @constraint(model, Y[n,j,k] <= psi_ub[n])
            end
        end
    end    
    for n in 1:N
        for j in 1:N
            @constraint(model, sum(Y[n,j,:]) == psi_3[n])
        end
    end

    @constraint(model, sum(X,dims=2) .== 1)
    @objective(model, Max, delta)

    optimize!(model)
    status = JuMP.termination_status(model)
    # println("status: ", status)
    # solution_summary(model)
    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        X_val = round.(value.(X))
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        X_val = ones(N,N) .* NaN
        solve_time = NaN
    end
    return obj_val,X_val,solve_time
end

function Solve_RO_one_side_exp_with_Price(N,N_u,A_lb,A_ub,B_lb,B_ub,u,price)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, delta)                           # 标量 δ
    @variable(model, omega_lb[1:N,1:N_u] >= 0)            
    @variable(model, omega_ub[1:N,1:N_u] >= 0)           
    @variable(model, pi_lb[1:N,1:N] >= 0)     
    @variable(model, pi_ub[1:N,1:N] >= 0)     

    # exponential variables
    @variable(model, psi_1[1:N])                   
    @variable(model, psi_2[1:N])                   
    @variable(model, psi_3[1:N])        
    
    for n in 1:N
        @constraint(model, omega_lb[n,:] .- omega_ub[n,:] .+ psi_3[n] .* u .== 0)
    end

    for n in 1:N
        @constraint(model, pi_lb[n,:] .- pi_ub[n,:] .+ psi_3[n] .* price .== 0)
    end

    @constraint(model, sum(omega_ub .* A_ub) - sum(omega_lb .* A_lb) + sum(pi_ub .* B_ub) - sum(pi_lb .* B_lb) + delta + sum(psi_2) <= 0)

    for n in 1:N
        @constraint(model, delta + psi_1[n] - price[n] <= 0)
    end

    for n in 1:N
        @constraint(model, [psi_3[n], psi_2[n], psi_1[n]] in MOI.DualExponentialCone())
    end

    @objective(model, Max, delta)

    optimize!(model)
    status = JuMP.termination_status(model)
    # println("status: ", status)
    # solution_summary(model)
    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        solve_time = NaN
    end
    return obj_val,solve_time
end