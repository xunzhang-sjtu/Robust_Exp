function Run_Oracle(iterations, N, N_u, K, Input_Data,print_flag)
    RST_ = Dict()
    for iter in 1:iterations
        Obs_Feat = Input_Data["iter=$(iter)_Obs_Feat"]
        A_true = Input_Data["iter=$(iter)_A_true"]
        B_true = Input_Data["iter=$(iter)_B_true"]
        P_dag = Input_Data["iter=$(iter)_P_dag"]
        
        obj_ETO,X_ETO,time_ETO = Solve_ETO(N,N_u,K,A_true,B_true,Obs_Feat,P_dag)
        rev_ETO, price_ETO = compute_oof(X_ETO, A_true, B_true, Obs_Feat, P_dag)
        if print_flag
            println("iter=$(iter): rev_Oracle = ",round(rev_ETO,digits=6),", price_Oracle = ",price_ETO)
        end
        RST_["iter=$(iter)_Time"] = time_ETO
        RST_["iter=$(iter)_Rev"] = rev_ETO
        RST_["iter=$(iter)_Price"] = price_ETO
        RST_["iter=$(iter)_Obj"] = obj_ETO
    end
    return RST_
end

function Run_ETO(iterations, N, N_u, K, Input_Data,print_flag)
    RST_ETO = Dict()
    for iter in 1:iterations
        Obs_Feat = Input_Data["iter=$(iter)_Obs_Feat"]
        A_hat = Input_Data["iter=$(iter)_A_hat"]
        B_hat = Input_Data["iter=$(iter)_B_hat"]
        A_true = Input_Data["iter=$(iter)_A_true"]
        B_true = Input_Data["iter=$(iter)_B_true"]
        P_dag = Input_Data["iter=$(iter)_P_dag"]
        
        obj_ETO,X_ETO,time_ETO = Solve_ETO(N,N_u,K,A_hat,B_hat,Obs_Feat,P_dag)
        rev_ETO, price_ETO = compute_oof(X_ETO, A_true, B_true, Obs_Feat, P_dag)
        if print_flag
            println("iter=$(iter): rev_ETO = ",round(rev_ETO,digits=6),", price_ETO = ",price_ETO)
        end
        RST_ETO["iter=$(iter)_Time"] = time_ETO
        RST_ETO["iter=$(iter)_Rev"] = rev_ETO
        RST_ETO["iter=$(iter)_Price"] = price_ETO
        RST_ETO["iter=$(iter)_Obj"] = obj_ETO
    end
    return RST_ETO
end

function Run_RO(RO_coef, iterations, N, N_u, K, Input_Data,psi_coef,model_name,print_flag)
    RST_RO = Dict()
    for iter in 1:iterations
        Obs_Feat = Input_Data["iter=$(iter)_Obs_Feat"]
        A_true = Input_Data["iter=$(iter)_A_true"]
        B_true = Input_Data["iter=$(iter)_B_true"]
        P_dag = Input_Data["iter=$(iter)_P_dag"]

        psi_lb = Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_psi_lb"]
        psi_ub = Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_psi_ub"]
        phi_lb = Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_phi_lb"]
        phi_ub = Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_phi_ub"]

        A_lb = Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_A_lb"]
        A_ub = Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_A_ub"]
        B_lb = Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_B_lb"]
        B_ub = Input_Data["iter=$(iter)_RO_coef=$(RO_coef)_B_ub"]
        if model_name == "Two_Side"
            obj_RO,X_RO,time_RO = Solve_RO(N,N_u,K,A_lb,A_ub,B_lb,B_ub,Obs_Feat,P_dag,psi_lb,psi_ub,phi_lb,phi_ub)
        else
            obj_RO,X_RO,time_RO = Solve_RO_one_side_exp(N,N_u,K,A_lb,A_ub,B_lb,B_ub,Obs_Feat,P_dag,psi_lb,psi_ub)
        end
        
        rev_RO, price_RO = compute_oof(X_RO, A_true, B_true, Obs_Feat, P_dag)
        if print_flag
            println("iter=$(iter): rev_RO = ",round(rev_RO,digits=6),", price_RO = ",price_RO)
        end
        RST_RO["iter=$(iter)_Time"] = time_RO
        RST_RO["iter=$(iter)_Rev"] = rev_RO
        RST_RO["iter=$(iter)_Price"] = price_RO
        RST_RO["iter=$(iter)_Obj"] = obj_RO
    end
    return RST_RO
end
