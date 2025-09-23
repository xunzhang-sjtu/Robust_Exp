function Product_Design_Ours_ETO(d,nu0, nu, r0, r, z_input)

    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, a0 >= 0)                       
    @variable(model, a1 >= 0) 
    @variable(model, u)                       
    @variable(model, v)
    @variable(model, x[1:d], Bin)
    @variable(model, y[1:d])
    @variable(model, z_int[1:d])

    @constraint(model, sum(x) >= 1)
    @constraint(model, a0 + a1 == 1)
    @constraint(model, [u,a0,a1] in MOI.ExponentialCone())
    @constraint(model, [-v,a1,a0] in MOI.ExponentialCone())
    @constraint(model, u == nu0 * a0 + nu' * z_int)
    @constraint(model, v == a1 * nu0 + nu' * y)
    for ind in 1:d
        @constraint(model, z_int[ind] <= a0)
        @constraint(model, z_int[ind] <= x[ind])
        @constraint(model, z_int[ind] >= x[ind] - (1 - a0))
        @constraint(model, z_int[ind] >= 0)
    end
    for ind in 1:d
        @constraint(model, y[ind] <= a1)
        @constraint(model, y[ind] <= x[ind])
        @constraint(model, y[ind] >= x[ind] - (1 - a1))
        @constraint(model, y[ind] >= 0)
    end
    @objective(model, Max,r0 * a1 + r' * y)
    optimize!(model)
    status = JuMP.termination_status(model)
    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        x_val = value.(x)
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        x_val = ones(d) .* NaN
        solve_time = NaN
    end
    return obj_val, x_val, solve_time
end

function Product_Design_ETO(d,nu0, nu, r0,r, z_input)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, a0 >= 0)                       
    @variable(model, a1 >= 0) 
    @variable(model, b0)                       
    @variable(model, b1) 
    @variable(model, v0)                       
    @variable(model, v1)
    @variable(model, u)                       
    @variable(model, w)
    @variable(model, x[1:d], Bin)
    @variable(model, y[1:d])
    @variable(model, phi)

    @constraint(model, a0 + a1 == 1)
    @constraint(model, w + b1 + b0 >= phi)
    @constraint(model, [b1,a1,1] in MOI.ExponentialCone())
    @constraint(model, [b0,a0,1] in MOI.ExponentialCone())
    @constraint(model, v0 + v1 <= 1)
    @constraint(model, [u-phi,1,v1] in MOI.ExponentialCone())
    @constraint(model, [-phi,1,v0] in MOI.ExponentialCone())
    @constraint(model, w == nu0 * a1 + nu' * y)
    @constraint(model, u == nu0 + nu' * x)
    for ind in 1:d
        @constraint(model, y[ind] <= a1)
        @constraint(model, y[ind] <= x[ind])
        @constraint(model, y[ind] >= x[ind] - (1 - a1))
        @constraint(model, y[ind] >= 0)
    end
    @constraint(model, sum(x) >= 1)
    @objective(model, Max,r0 * a1 + r' * y)

    optimize!(model)
    status = JuMP.termination_status(model)

    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        x_val = value.(x)
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        x_val = ones(d) .* NaN
        solve_time = NaN
    end
    return obj_val, x_val, solve_time
end




function Robust_Product_Design(N, n_x, n_c, gamma, psi_lb, psi_ub, phi_lb, phi_ub, alp0, alp, beta, A, r0, r, z_input)

    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, delta)                           # 标量 δ
    @variable(model, eta[1:N])  
    @variable(model, omega_0[1:N])    
    @variable(model, omega[1:N,1:n_x])  
    @variable(model, pi_[1:N,1:n_c])
    @variable(model, D_[1:N,1:n_x,1:n_c])  
    @variable(model, x[1:n_x], Bin)           
    @variable(model, psi[1:3])                        
    @variable(model, phi[1:3])             
    @variable(model, y[1:n_x])        
    @variable(model, z[1:n_x])  

    @constraint(model, omega_0[1] + psi[3] == 0)
    @constraint(model, omega[1,:] .+ y .== 0)
    for ind in 1:n_x
        @constraint(model, y[ind] >= psi_lb * x[ind])
        @constraint(model, y[ind] <= psi_ub * x[ind])
        @constraint(model, y[ind] >= psi[3] - psi_ub * (1 - x[ind]))
        @constraint(model, y[ind] <= psi[3] - psi_lb * (1 - x[ind]))
    end
    @constraint(model, pi_[1,:] .+  psi[3] .* z_input.== 0)
    for i in 1:n_c
        @constraint(model, D_[1,:,i] .+ y.* z_input[i] .== 0)
    end

    @constraint(model, omega_0[2] - phi[3] == 0)
    @constraint(model, omega[2,:] .- z .== 0)
    for ind in 1:n_x
        @constraint(model, z[ind] >= phi_lb * x[ind])
        @constraint(model, z[ind] <= phi_ub * x[ind])
        @constraint(model, z[ind] >= phi[3] - phi_ub * (1 - x[ind]))
        @constraint(model, z[ind] <= phi[3] - phi_lb * (1 - x[ind]))
    end
    @constraint(model, pi_[2,:] .- phi[3] .* z_input .== 0)
    for i in 1:n_c
        @constraint(model, D_[2,:,i] .- z.* z_input[i] .== 0)
    end

    @constraint(model, delta + psi[2] + phi[1] + eta[1] * gamma - alp0 * omega_0[1] - alp' * omega[1,:] - beta' * pi_[1,:] - sum(A .* D_[1,:,:]) <= 0)
    @constraint(model, delta + psi[1] + phi[2] + eta[2] * gamma - alp0 * omega_0[2] - alp' * omega[2,:] - beta' * pi_[2,:] - sum(A .* D_[2,:,:]) <= r0 + r' * x)

    @constraint(model, [psi[3], psi[2], psi[1]] in MOI.DualExponentialCone())
    @constraint(model, [phi[3], phi[2], phi[1]] in MOI.DualExponentialCone())

    @constraint(model, [eta[1];omega_0[1];omega[1,:];pi_[1,:];vec(D_[1,:,:])] in SecondOrderCone())
    @constraint(model, [eta[2];omega_0[2];omega[2,:];pi_[2,:];vec(D_[2,:,:])] in SecondOrderCone())

    @constraint(model, sum(x) >= 1)
    @objective(model, Max, delta)
    optimize!(model)

    status = JuMP.termination_status(model)
    # println("status: ", status)
    # solution_summary(model)
    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        x_val = value.(x)
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        x_val = ones(n_x) .* NaN
        solve_time = NaN
    end
    return obj_val, x_val, solve_time
end

