function search_opt_price(N,p_lb,p_ub,b_n)
    model = Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", true)
    # 定义变量
    @variable(model, price[1:N])                      # y_{nk}
    @constraint(model, price .>= p_lb)
    @constraint(model, price .<= p_ub)
    @objective(model, Max,b_n' * price )
    optimize!(model)
    obj_val = objective_value(model)
    return obj_val
end


function generate_Input_Data(S_train,S_test,iterations, N, N_u, K, offdiag_sign,max_offdiag,P_bar)
    Input_Data = Dict()
    for iter in 1:iterations
        A_true, B_true = Generate_Coef(N_u, N,max_offdiag,offdiag_sign);
        U_train, P_train = Generate_Feat_Data(N_u, N, S_train);
        U_test, P_test = Generate_Feat_Data(N_u, N, S_test);

        Input_Data["iter=$(iter)_Obs_Feat"] = U_test[1,:];
        Input_Data["iter=$(iter)_A_true"] = A_true;
        Input_Data["iter=$(iter)_B_true"] = B_true;
        Input_Data["iter=$(iter)_P_dag"] = round.(rand(N, K) .* P_bar; digits=2);
        Input_Data["iter=$(iter)_U_train"] = U_train;
        Input_Data["iter=$(iter)_P_train"] = P_train;

        A_hat, B_hat = Estimate_MNL_Para(U_train, P_train, S_train, N, N_u, N, A_true, B_true);

        Input_Data["iter=$(iter)_A_hat"] = A_hat
        Input_Data["iter=$(iter)_B_hat"] = B_hat
    end
    return Input_Data
end

function Calculate_Hyper_Param(RO_coef_all, iterations, N, N_u, K, Input_Data)
    for iter in 1:iterations
        Obs_Feat = Input_Data["iter=$(iter)_Obs_Feat"]
        # A_true = Input_Data["iter=$(iter)_A_true"]
        # B_true = Input_Data["iter=$(iter)_B_true"]
        P_dag = Input_Data["iter=$(iter)_P_dag"]
        A_hat = Input_Data["iter=$(iter)_A_hat"]
        B_hat = Input_Data["iter=$(iter)_B_hat"]

        for RO_coef in RO_coef_all
            A_lb = A_hat .- RO_coef .* abs.(A_hat);
            A_ub = A_hat .+ RO_coef .* abs.(A_hat);
            B_lb = B_hat .- RO_coef .* abs.(B_hat);
            B_ub = B_hat .+ RO_coef .* abs.(B_hat);
            
            p_ub = vec(maximum(P_dag,dims=2))
            p_lb = vec(minimum(P_dag,dims=2))
            p_max = maximum(p_ub)
            p_min = minimum(p_lb)
            Obs_Feat_Trun = [max(-Obs_Feat[ind],0) for ind in 1:N_u]
            psi_lb = zeros(N)
            for n in 1:N
                b_n = B_lb[n,:]
                obj_n = search_opt_price(N,p_lb,p_ub,b_n)
                psi_lb[n] = max(-10000,-exp(-Obs_Feat_Trun' * (A_ub[n,:] - A_lb[n,:]) + Obs_Feat' * A_lb[n,:] + obj_n)*(p_max-p_min))
            end        

            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_psi_lb"] = psi_lb
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_psi_ub"] = zeros(N)
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_phi_lb"] = [p_min - p_max for i in 1:N]
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_phi_ub"] = zeros(N)

            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_A_lb"] = A_lb
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_A_ub"] = A_ub
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_B_lb"] = B_lb
            Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_B_ub"] = B_ub
        end
    end
    return Input_Data
end