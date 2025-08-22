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