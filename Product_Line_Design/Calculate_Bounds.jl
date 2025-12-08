# function compute_r_bounds(r,index_given,N,N_x,Time_Limit,rev_gap,c_l,d_r,model_sense)
#     model = Model(Mosek.Optimizer)
#     set_attribute(model, "QUIET", true)
#     set_optimizer_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", Time_Limit) 
#     # 定义变量
        
#     @variable(model, X[1:N, 1:N_x], Bin)        # 二进制变量 x_{jk}
#     @constraint(model, X * c_l .>= d_r)

#     for ind1 in 1:(N-1)
#         @constraint(model, r' * X[ind1,:] >= r' * X[(ind1+1),:] + rev_gap)
#     end
#     if model_sense == "Max"
#         @objective(model, Max, r' * X[index_given,:])
#     elseif model_sense == "Min"
#         @objective(model, Min, r' * X[index_given,:])
#     end

#     optimize!(model)
#     status = JuMP.termination_status(model)
#     # println("status: ", status)
#     # solution_summary(model)
#     if status == MOI.OPTIMAL  || status == MOI.TIME_LIMIT
#         sol_status = string(status)
#         obj_val = objective_value(model)
#         X_val = round.(value.(X))
#         solve_time = JuMP.solve_time(model)
#     else
#         sol_status = "Others"
#         obj_val = NaN
#         X_val = ones(N,N) .* NaN
#         solve_time = NaN
#     end
#     return sol_status, obj_val, X_val, solve_time
# end

# r = r_params.r
# phi_lb = zeros(N)
# for n in 1:N 
#     sol_status, obj_min, X_val, solve_time = compute_r_bounds(r,n,N,N_x,Time_Limit,rev_gap,c_l,d_r,"Min")
#     sol_status, obj_max, X_val, solve_time = compute_r_bounds(r,n,N,N_x,Time_Limit,rev_gap,c_l,d_r,"Max")
#     phi_lb[n] = obj_min - obj_max
# end