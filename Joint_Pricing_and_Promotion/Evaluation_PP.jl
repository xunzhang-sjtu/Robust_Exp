function compute_oof(X_given, A, B, u, p_dag)
    N = length(A[:,1])
    if any(isnan, X_given)
        rev = NaN
        price = ones(N) .* NaN
    else
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
    end
    return sum(rev),price
end


function compute_oof_compact(price_indices, promo,w,P_dag)
    price = sum(price_indices .* P_dag,dims=2)[:,1]
    N_w = length(w)
    X = zeros(N,N_w)

    for n in 1:N 
        all_indices = collect(1:N*N)
        this_indices = collect((n-1)*N + 1:(n-1)*N + 3)
        diff_indices = setdiff(all_indices, this_indices)
        # println("n=$n,this_indices=",this_indices,",diff_indices=",diff_indices)
        X[n,this_indices[1]] = 1.0
        X[n,this_indices[2]] = promo[n]
        X[n,this_indices[3]] = price[n]
    end

    utilities = exp.(X * w)
    prob = utilities ./ (1 + sum(utilities))
    rev = sum(prob .* price)
    return rev,price
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

function greedy_Compact(K,w,P_dag)
    N_w = length(w)
    u_set = [[1,0,0],[0,1,0],[0,0,1]]
    for u in u_set
        x1 = zeros(N_w)
        x1[1] = 1.0
        x1[2] = u[1]

        x2 = zeros(N_w)
        x2[4] = 1.0
        x2[5] = u[2]

        x3 = zeros(N_w)
        x3[7] = 1.0
        x3[8] = u[3]

        rev_opt = 0
        price_opt = zeros(N)
        for n in 1:K
            p1 = P_dag[1,n]
            for j in 1:K
                p2 = P_dag[2,j]
                for l in 1:K
                    p3 = P_dag[3,l]
                    x1[3] = p1
                    x2[6] = p2
                    x3[9] = p3

                    utilities = [1,exp(w' * x1),exp(w' * x2),exp(w' * x3)]
                    prob = utilities ./ sum(utilities)
                    rev = p1 * prob[2] + p2 * prob[3] + p3 * prob[4]

                    if rev > rev_opt
                        rev_opt = rev
                        price_opt = [p1,p2,p3] 
                    end
                end
            end
        end
        println("promotion: ", u)
        println("Optimal revenue by greedy: ", rev_opt)
        println("Optimal prices by greedy: ", price_opt)
    end

end