using JuMP
using MosekTools
# using COPT

function Solve_ETO(N,N_u,K,A,B,u,p_dag)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # model = Model(COPT.ConeOptimizer)
    # 定义变量
    @variable(model, rho_0 >= 0)                       # ρ₀ ≥ 0
    @variable(model, rho[1:N] >= 0)                    # ρ_n ≥ 0
    @variable(model, v_sigma[1:N_u])                   # ς = ρ₀ * u
    @variable(model, v_phi[1:N])                       # φ_n
    @variable(model, v_theta[1:N,1:N_u])               # ς = ρ₀ * u
    @variable(model, v_pi[1:N,1:N])  
    @variable(model, Y[1:N, 1:K])                      # y_{nk}
    @variable(model, Z[1:N, 1:K])                      # z_{nk}
    @variable(model, X[1:N, 1:K], Bin)                 # x_{nk} ∈ {0,1}
    @variable(model, W[1:N,1:N,1:K])                   # z_{nk}
    
    @constraint(model, rho_0 + sum(rho) == 1)

    for n in 1:N
        @constraint(model, [A[n,:]' * v_sigma + B[n,:]' * v_phi,rho_0,rho[n]] in MOI.ExponentialCone())
    end

    @constraint(model, v_sigma .== rho_0 .* u)
    @constraint(model, v_phi .== sum(Z .* p_dag, dims=2))
    for n in 1:N
        for k in 1:K
            # z_{nk} bounds
            @constraint(model, 0 <= Z[n, k])
            @constraint(model, Z[n, k] <= X[n, k])
            @constraint(model, Z[n, k] >= rho_0 - (1 - X[n, k]))
            @constraint(model, Z[n, k] <= rho_0)
        end
    end

    for n in 1:N
        @constraint(model, [-A[n,:]' * v_theta[n,:] - B[n,:]' * v_pi[n,:], rho[n], rho_0] in MOI.ExponentialCone())
    end
    for n in 1:N
        @constraint(model, v_theta[n,:] .== rho[n] .* u)
    end
    for n in 1:N
        @constraint(model, v_pi[n,:] .== sum(W[n,:,:] .* p_dag, dims=2))
    end
    for n in 1:N
        for j in 1:N
            for k in 1:K
                # z_{nk} bounds
                @constraint(model, 0 <= W[n,j, k])
                @constraint(model, W[n,j,k] <= X[j, k])
                @constraint(model, W[n,j,k] >= rho[n] - (1 - X[j, k]))
                @constraint(model, W[n,j, k] <= rho[n])
            end
        end
    end

    # objective 
    for n in 1:N
        for k in 1:K
            # y_{nk} bounds
            @constraint(model, 0 <= Y[n, k])
            @constraint(model, Y[n, k] <= X[n, k])
            @constraint(model, Y[n, k] >= rho[n] - (1 - X[n, k]))
            @constraint(model, Y[n, k] <= rho[n])
        end
    end

    @constraint(model, sum(X,dims=2) .== 1)

    @objective(model, Max,sum(Y .* p_dag))

    optimize!(model)
    status = JuMP.termination_status(model)
    # solution_summary(model)
    if status == MOI.OPTIMAL
        obj_val = objective_value(model)
        X_val = value.(X)
        solve_time = JuMP.solve_time(model)
    else
        obj_val = NaN
        X_val = ones(N,N) .* NaN
        solve_time = NaN
    end
    return obj_val, X_val,solve_time
end

# function Solve_ETO_COPT_Version(N,N_u,K,A,B,u,p_dag)
#     # model = Model(Mosek.Optimizer)
#     # set_attribute(model, "QUIET", true)
#     model = Model(COPT.ConeOptimizer)
#     # 定义变量
#     @variable(model, rho_0 >= 0)                       # ρ₀ ≥ 0
#     @variable(model, rho[1:N] >= 0)                    # ρ_n ≥ 0
#     @variable(model, v_sigma[1:N_u])                   # ς = ρ₀ * u
#     @variable(model, v_phi[1:N])                       # φ_n
#     @variable(model, v_theta[1:N,1:N_u])               # ς = ρ₀ * u
#     @variable(model, v_pi[1:N,1:N])  
#     @variable(model, Y[1:N, 1:K])                      # y_{nk}
#     @variable(model, Z[1:N, 1:K])                      # z_{nk}
#     @variable(model, X[1:N, 1:K], Bin)                 # x_{nk} ∈ {0,1}
#     @variable(model, W[1:N,1:N,1:K])                   # z_{nk}
    

#     for n in 1:N
#         for k in 1:K
#             # y_{nk} bounds
#             @constraint(model, 0 <= Y[n, k])
#             @constraint(model, Y[n, k] <= X[n, k])
#             @constraint(model, Y[n, k] >= rho[n] - (1 - X[n, k]))
#             @constraint(model, Y[n, k] <= rho[n])
#         end
#     end

#     @constraint(model, rho_0 + sum(rho) == 1)

#     for n in 1:N
#         @constraint(model, [A[n,:]' * v_sigma + B[n,:]' * v_phi,rho_0,rho[n]] in MOI.ExponentialCone())
#     end
#     @constraint(model, v_sigma .== rho_0 .* u)
#     @constraint(model, v_phi .== sum(Z .* p_dag, dims=2))
#     for n in 1:N
#         for k in 1:K
#             # z_{nk} bounds
#             @constraint(model, 0 <= Z[n, k])
#             @constraint(model, Z[n, k] <= X[n, k])
#             @constraint(model, Z[n, k] >= rho_0 - (1 - X[n, k]))
#             @constraint(model, Z[n, k] <= rho_0)
#         end
#     end

#     for n in 1:N
#         @constraint(model, [-A[n,:]' * v_theta[n,:] - B[n,:]' * v_pi[n,:], rho[n], rho_0] in MOI.ExponentialCone())
#     end
#     for n in 1:N
#         @constraint(model, v_theta[n,:] .== rho[n] .* u)
#     end
#     for n in 1:N
#         @constraint(model, v_pi[n,:] .== sum(W[n,:,:] .* p_dag, dims=2))
#     end
#     for n in 1:N
#         for j in 1:N
#             for k in 1:K
#                 # z_{nk} bounds
#                 @constraint(model, 0 <= W[n,j, k])
#                 @constraint(model, W[n,j,k] <= X[j, k])
#                 @constraint(model, W[n,j,k] >= rho[n] - (1 - X[j, k]))
#                 @constraint(model, W[n,j, k] <= rho[n])
#             end
#         end
#     end

#     @constraint(model, sum(X,dims=2) .== 1)

#     @objective(model, Max,sum(Y .* p_dag))

#     optimize!(model)
#     status = JuMP.termination_status(model)
#     # solution_summary(model)
#     if status == MOI.OPTIMAL
#         obj_val = objective_value(model)
#         X_val = value.(X)
#         solve_time = JuMP.solve_time(model)
#     else
#         obj_val = NaN
#         X_val = ones(N,N) .* NaN
#         solve_time = NaN
#     end
#     return obj_val, X_val,solve_time
# end