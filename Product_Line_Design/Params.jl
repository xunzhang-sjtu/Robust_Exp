function set_Params(N, N_x, N_u, N_nonzero, S_train, S_test, m,c_l,d_r,rev_gap,Time_Limit,dual_norm,gamma_list,num_c,instances)

    Params = Dict()
    # --- Parameters for data generation ---
    Params["N"] = N
    N = 3 # num of product
    N_x = 8 # num of product feature
    c_l = ones(N_x)  # X * c_l >= d_r
    d_r = ones(N) * 2
    rev_gap = 0.00001
    N_u = 1 # num of customer feature
    S_train = 100 # num of training samples
    S_test = 1 # num of training samples
    m = 5 # num of candidates in training samples
    N_nonzero = 5 # num of nonzero entries in A
    Time_Limit = 300 # time limit for each instance in seconds
    dual_norm = 2
    gamma_list = [0.0,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0]

    psi_lb = -2 * ones(N)
    psi_ub = 0 * ones(N)
    phi_lb = -2 * ones(N)
    phi_ub = 0 * ones(N)

    num_c = 4
    instances = 100
end