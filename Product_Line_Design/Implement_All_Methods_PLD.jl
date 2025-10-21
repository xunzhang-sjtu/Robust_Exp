function solve_ETO_This(S_test,N,N_x,theta_Input,theta_true,r_params,c_l,d_r,rev_gap,num_c,Time_Limit,Z_test)
    r0 = r_params.r0; r = r_params.r;

    RST_Dict = Dict()
    obj_list = zeros(S_test); x_list = zeros(S_test,N,N_x); time_list = zeros(S_test); profit_list = zeros(S_test); status_list = Vector{String}(undef, S_test)
    status_all = "OPTIMAL"
    for i in 1:S_test
        z_input = Z_test[i,:];
        nu0, nu = compute_w(theta_Input,z_input)
        obj_list[i], x_list[i,:,:], time_list[i],status_list[i] = ETO_PLD(N,N_x,nu0, nu, r0, r,c_l,d_r,rev_gap,num_c,Time_Limit);
        if status_list[i] != "OPTIMAL"
            status_all = status_list[i]
            println("Warning: The optimization for test instance $i did not reach optimality. Status: ", status_list[i])
            break
        end
        profit_list[i] = calculate_profit(theta_true, r0, r, x_list[i,:,:], z_input)                       
    end
    RST_Dict["obj"] = obj_list
    RST_Dict["X"] = x_list
    RST_Dict["time"] = time_list
    RST_Dict["profit"] = profit_list
    RST_Dict["status"] = status_list
    return RST_Dict,status_all
end

function solve_RO_this(S_test,N,N_x,theta_Input,theta_true,r_params,c_l,d_r,rev_gap,num_c,Time_Limit,Z_test,gamma,psi_lb,psi_ub,phi_lb,phi_ub)
    r0 = r_params.r0; r = r_params.r;
    RST_Dict = Dict();
    
    obj_list = zeros(S_test); x_list = zeros(S_test,N,N_x); time_list = zeros(S_test); profit_list = zeros(S_test); status_list = Vector{String}(undef, S_test)
    status_all = "OPTIMAL"
    for i in 1:S_test
        z_input = Z_test[i,:];                
        nu0, nu = compute_w(theta_Input,z_input)
        obj_list[i], x_list[i,:,:], time_list[i],status_list[i] = RO_PLD(N,N_x,nu0,nu,r0,r,c_l,d_r,rev_gap,psi_lb,psi_ub,phi_lb,phi_ub,gamma,dual_norm,num_c,Time_Limit)
        if status_list[i] != "OPTIMAL"
            status_all = status_list[i]
            println("Warning: The RO optimization for test instance $i did not reach optimality. Status: ", status_list[i])
            break
        end
        profit_list[i] = calculate_profit(theta_true, r0, r, x_list[i,:,:], z_input)
    end
    RST_Dict["obj"] = obj_list
    RST_Dict["X"] = x_list
    RST_Dict["time"] = time_list
    RST_Dict["profit"] = profit_list
    RST_Dict["status"] = status_list
    return RST_Dict,status_all
end