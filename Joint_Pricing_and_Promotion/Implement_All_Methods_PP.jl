function Run_Oracle(iterations, N, N_u, K, Input_Data,print_flag)
    RST_ = Dict()
    for iter in 1:iterations
        A_true = Input_Data["iter=$(iter)_A_true"]
        B_true = Input_Data["iter=$(iter)_B_true"]
        P_dag = Input_Data["iter=$(iter)_P_dag"]

        obj_ETO,X_ETO,Promo_ETO, time_ETO = Solve_ETO(N,N_u,K,A_true, B_true,P_dag)
        rev_ETO, price_ETO = compute_oof(X_ETO, A_true, B_true, vcat(Promo_ETO,1), P_dag)
        if print_flag
            println("iter=$(iter): rev_Oracle = ",round(rev_ETO,digits=6),", price_Oracle = ",price_ETO)
        end
        RST_["iter=$(iter)_Time"] = time_ETO
        RST_["iter=$(iter)_Rev"] = rev_ETO
        RST_["iter=$(iter)_Price"] = price_ETO
        RST_["iter=$(iter)_Promo"] = Promo_ETO
        RST_["iter=$(iter)_Obj"] = obj_ETO
    end
    return RST_
end

function Run_ETO(iterations, N, N_u, K, Input_Data,print_flag)
    RST_ = Dict()
    for iter in 1:iterations
        A_true = Input_Data["iter=$(iter)_A_true"]
        B_true = Input_Data["iter=$(iter)_B_true"]
        A_hat = Input_Data["iter=$(iter)_A_hat"]
        B_hat = Input_Data["iter=$(iter)_B_hat"]
        P_dag = Input_Data["iter=$(iter)_P_dag"]

        obj_ETO,X_ETO,Promo_ETO, time_ETO = Solve_ETO(N,N_u,K,A_hat, B_hat,P_dag)
        rev_ETO, price_ETO = compute_oof(X_ETO, A_true, B_true, vcat(Promo_ETO,1), P_dag)
        if print_flag
            println("iter=$(iter): rev_ETO = ",round(rev_ETO,digits=6),", price_ETO = ",price_ETO)
        end
        RST_["iter=$(iter)_Time"] = time_ETO
        RST_["iter=$(iter)_Rev"] = rev_ETO
        RST_["iter=$(iter)_Price"] = price_ETO
        RST_["iter=$(iter)_Promo"] = Promo_ETO
        RST_["iter=$(iter)_Obj"] = obj_ETO
    end
    return RST_
end

function Run_RO(gamma_list, dual_norm,iterations, N, N_u, K, Input_Data,print_flag,psi_lb_coef,phi_lb_coef)
    RST_RO = Dict()
    for iter in 1:iterations
        A_true = Input_Data["iter=$(iter)_A_true"]
        B_true = Input_Data["iter=$(iter)_B_true"]
        A_hat = Input_Data["iter=$(iter)_A_hat"]
        B_hat = Input_Data["iter=$(iter)_B_hat"]
        P_dag = Input_Data["iter=$(iter)_P_dag"]

        psi_lb = psi_lb_coef * ones(N) 
        psi_ub = 0.0 * ones(N) 
        phi_lb = phi_lb_coef * ones(N) 
        phi_ub = 0.0 * ones(N) 
        RST_this = Dict()
        for gamma in gamma_list
            obj_RO,X_RO,Promo_RO, time_RO = Solve_RO(N,N_u,K,A_hat,B_hat,P_dag,psi_lb,psi_ub,phi_lb,phi_ub,gamma * ones(N),dual_norm)
            rev_RO, price_RO = compute_oof(X_RO, A_true, B_true, vcat(Promo_RO,1), P_dag)
            RST_this["obj_gamma=$(gamma)"] = obj_RO
            RST_this["price_gamma=$(gamma)"] = price_RO
            RST_this["Promo_gamma=$(gamma)"] = Promo_RO
            RST_this["time_gamma=$(gamma)"] = time_RO
            RST_this["Rev_gamma=$(gamma)"] = rev_RO
            # if print_flag
            #     println("iter=$(iter),gamma=$(gamma): rev_RO = ",round(rev_RO,digits=6),", price_RO = ",price_RO)
            # end
        end
        if print_flag
            println("***** iter=$(iter) *******")
        end
        RST_RO["iter=$(iter)_RST"] = RST_this
    end
    return RST_RO
end
