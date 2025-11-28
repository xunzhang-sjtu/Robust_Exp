function get_default_params_PP()
    Params = Dict()
    Params["N"] = 3
    Params["N_u"] = 1
    Params["K"] = 30
    Params["S_train"] = 100
    Params["S_test"] = 1000
    Params["P_bar"] = 30.0
    Params["iterations"] = 100
    Params["offdiag_sign"] = "mix"
    Params["max_offdiag"] = 1.0
    Params["Time_Limit"] = 180.0
    Params["gamma_list"] = [0.0, 0.005,0.01, 0.0125,0.015,0.0175,0.02]
    Params["dual_norm"] = 2
    Params["psi_lb_coef"] = -30.0
    Params["phi_lb_coef"] = -30.0
    Params["psi_lb"] = Params["psi_lb_coef"] * ones(Params["N"]) 
    Params["psi_ub"] = 0.0 * ones(Params["N"]) 
    Params["phi_lb"] = Params["phi_lb_coef"] * ones(Params["N"]) 
    Params["phi_ub"] = 0.0 * ones(Params["N"]) 
    Params["is_original_setting"] = true
    Params["seed"] = 3
    return Params
end
