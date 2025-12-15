function get_default_params_PLD()
    Params = Dict()
    # --- Parameters for data generation ---
    Params["N"] = 3
    Params["N_x"] = 8
    Params["c_l"] = ones(Params["N_x"])
    Params["d_r"] = ones(Params["N"]) * 2
    Params["rev_gap"] = 0.001
    Params["N_u"] = 1
    Params["S_train"] = 100
    Params["S_test"] = 1
    Params["N_Max"] = 5
    Params["N_nonzero"] = 5
    Params["Time_Limit"] = 300
    Params["dual_norm"] = 2
    Params["gamma_list"] = [0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0,5.0]
    Params["gamma_list_Wang_Qi_Max"] = [0.0,0.01,0.02,0.04,0.06,0.08]
    Params["psi_lb"] = -30 * ones(Params["N"])
    Params["psi_ub"] = 0 * ones(Params["N"])
    Params["phi_lb"] = -30 * ones(Params["N"])
    Params["phi_ub"] = 0 * ones(Params["N"])
    Params["num_c"] = 4
    Params["instances"] = 100   
    Params["seed"] = 1

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


    Params["coef_this_backup1"] = (alp0_lb=0.1, alp0_ub=0.2, 
                            alp_lb=-1.0, alp_ub=0.0, 
                            beta_lb=-0.2, beta_ub=0.2, 
                            A_lb=-0.2, A_ub=0.2, 
                            r0_lb=0.0, r0_ub=1.0, 
                            r_lb=0.0, r_ub=0.1); # 5% improvement over backup
    
    Params["coef_this_backup2"] = (alp0_lb=1.0, alp0_ub=2.0, 
                        alp_lb=-1.0, alp_ub=0.0, 
                        beta_lb=-0.2, beta_ub=0.2, 
                        A_lb=-0.2, A_ub=0.2, 
                        r0_lb=0.0, r0_ub=1.0, 
                        r_lb=0.0, r_ub=0.1); # 2% improvement over backup
    
    Params["coef_this_backup3"] = (alp0_lb=1.0, alp0_ub=2.0, 
                        alp_lb=-1.0, alp_ub=1.0, 
                        beta_lb=-0.2, beta_ub=0.2, 
                        A_lb=-0.2, A_ub=0.2, 
                        r0_lb=0.0, r0_ub=1.0, 
                        r_lb=0.0, r_ub=0.1); # No improvement over backup, because alpha can be positive

    Params["coef_backup4"] = (alp0_lb=0.01, alp0_ub=0.02, 
                            alp_lb=-1.0, alp_ub=0.0, 
                            beta_lb=-0.02, beta_ub=0.02, 
                            A_lb=-0.02, A_ub=0.02, 
                            r0_lb=0.0, r0_ub=1.0, 
                            r_lb=-1.0, r_ub=1.0);

    Params["coef_this"] = (alp0_lb=1.0, alp0_ub=2.0, 
                            alp_lb=-1.0, alp_ub=1.0, 
                            beta_lb=-2.0, beta_ub=2.0, 
                            A_lb=-2.0, A_ub=2.0, 
                            r0_lb=0.0, r0_ub=1.0,
                            r_lb=-1.0, r_ub=1.0);

    return Params
end