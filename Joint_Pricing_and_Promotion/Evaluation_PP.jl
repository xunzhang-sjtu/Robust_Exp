function compute_oof(X_given, A, B, u, p_dag)
    price = sum(X_given .* p_dag,dims=2)[:,1]
    utilities = ones(N+1)
    utilities[1:N] = exp.(vec(A * u + B * price))

    prob = zeros(N)
    rev = zeros(N)
    for i in 1:N
        prob[i] = utilities[i] / sum(utilities)
        rev[i] = prob[i] * price[i]
    end
    # println("prob = ", prob)
    # println("rev = ", rev)
    # println("total rev = ", sum(rev))
    return sum(rev),price
end

function greedy_search_optimal_price(N, K, A, B, P_dag)
    # only works for N = 2 and N_u = 1
    if N != 2
        error("This function only supports N = 3")
    end
    rev_opt = 0
    price_opt = zeros(N)
    for i in 1:K
        p1 = P_dag[1,i]
        for j in 1:K
            p2 = P_dag[2,j]
            price = [p1,p2]
            u1 = [1,0,1]
            utilities = ones(N+1)
            utilities[1:N] = exp.(vec(A * u1 + B * price))
            prob = zeros(N)
            rev = zeros(N)
            for i in 1:N
                prob[i] = utilities[i] / sum(utilities)
                rev[i] = prob[i] * price[i]
            end
            if sum(rev) > rev_opt
                rev_opt = sum(rev)
                price_opt = price 
            end

            u1 = [0,1,1]
            utilities = ones(N+1)
            utilities[1:N] = exp.(vec(A * u1 + B * price))
            prob = zeros(N)
            rev = zeros(N)
            for i in 1:N
                prob[i] = utilities[i] / sum(utilities)
                rev[i] = prob[i] * price[i]
            end
            if sum(rev) > rev_opt
                rev_opt = sum(rev)
                price_opt = price 
            end
        end
    end
    println("Optimal revenue = ", rev_opt)
    println("Optimal price = ", price_opt)
end