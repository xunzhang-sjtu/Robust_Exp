function get_default_params_PLD()
    Params = Dict()
    # --- Parameters for data generation ---
    Params["N"] = 3
    Params["N_x"] = 8
    Params["c_l"] = ones(Params["N_x"])
    Params["d_r"] = ones(Params["N"]) * 2
    Params["rev_gap"] = 0.001
    Params["N_u"] = 10
    Params["S_train_all"] = [100,200,500]
    Params["S_test"] = 1
    Params["N_Max"] = 5
    Params["N_nonzero"] = 20
    Params["Time_Limit"] = 600
    Params["dual_norm"] = 2
    Params["norm_bounds"] = 20
    Params["gamma_list"] = [0.0,0.1,0.2,0.4,0.6,0.8,1.0]
    Params["gamma_list_Wang_Qi_Max"] = [0.0,0.01,0.02,0.04,0.06,0.08]
    Params["psi_lb"] = -5 * ones(Params["N"])
    Params["psi_ub"] = 0 * ones(Params["N"])
    Params["phi_lb"] = -5 * ones(Params["N"])
    Params["phi_ub"] = 0 * ones(Params["N"])
    Params["num_c"] = 8
    Params["is_ridge"] = true
    Params["lambda_all"] = [0.01]
    Params["instances"] = 50
    Params["seed"] = 2

    Params["coef_backup"] = (alp0_lb=0.01, alp0_ub=0.02, 
                            alp_lb=-1.0, alp_ub=0.0, 
                            beta_lb=-0.02, beta_ub=0.02, 
                            A_lb=-0.02, A_ub=0.02, 
                            r0_lb=0.0, r0_ub=1.0, 
                            r_lb=0.0, r_ub=0.1);

    Params["coef_Wang_Qi_Shen"] = (alp0_lb=1.0, alp0_ub=2.0, 
                            alp_lb=-1.0, alp_ub=1.0, 
                            beta_lb=-2.0, beta_ub=2.0, 
                            A_lb=-2.0, A_ub=2.0, 
                            r0_lb=0.0, r0_ub=1.0,
                            r_lb=-1.0, r_ub=1.0);

    Params["coef_this"] = (alp0_lb=1.0, alp0_ub=2.0, 
                            alp_lb=-1.0, alp_ub=0.0, 
                            beta_lb=-2.0, beta_ub=2.0, 
                            A_lb=-2.0, A_ub=2.0, 
                            r0_lb=0.0, r0_ub=1.0, 
                            r_lb=0.0, r_ub=0.1);
    return Params
end


function get_Wang_Qi_Shen_params_PLD()
    Params = Dict()
    # --- Parameters for data generation ---
    Params["N"] = 3
    Params["N_x"] = 8
    Params["c_l"] = ones(Params["N_x"])
    Params["d_r"] = ones(Params["N"]) * 2
    Params["rev_gap"] = 0.001
    Params["N_u"] = 10
    Params["S_train_all"] = [100]
    Params["S_test"] = 1
    Params["N_Max"] = 5
    Params["N_nonzero"] = 10
    Params["Time_Limit"] = 300
    Params["dual_norm"] = 2
    Params["norm_bounds"] = 20
    Params["gamma_list"] = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]
    Params["psi_lb"] = -5 * ones(Params["N"])
    Params["psi_ub"] = 0 * ones(Params["N"])
    Params["phi_lb"] = -5 * ones(Params["N"])
    Params["phi_ub"] = 0 * ones(Params["N"])
    Params["num_c"] = 4
    Params["is_ridge"] = true
    Params["lambda_all"] = [0.01]
    Params["instances"] = 20
    Params["seed"] = 12
    Params["coef_this"] = (alp0_lb=1.0, alp0_ub=2.0, 
                            alp_lb=-1.0, alp_ub=1.0, 
                            beta_lb=-2.0, beta_ub=2.0, 
                            A_lb=-2.0, A_ub=2.0, 
                            r0_lb=0.0, r0_ub=1.0,
                            r_lb=-1.0, r_ub=1.0);
          
    return Params
end