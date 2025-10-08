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

function greedy_search_optimal_price(N, K, A, B, P_dag,u)
    # only works for N = 3
    if N != 3
        error("This function only supports N = 3")
    end
    if N == 3 
        rev_opt = 0
        price_opt = zeros(N)

        rev_min = 1000
        price_min = zeros(N)
        for i in 1:K
            p1 = P_dag[1,i]
            for j in 1:K
                p2 = P_dag[2,j]
                for l in 1:K
                    p3 = P_dag[3,l]
                    price = [p1,p2,p3]
                    utilities = ones(N+1)
                    utilities[1:N] = exp.(vec(A * u + B * price))
                
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

                    if sum(rev) < rev_min
                        rev_min = sum(rev)
                        price_min = price
                    end
                end
            end
        end
        println("Optimal revenue = ", rev_opt)
        println("Optimal price = ", price_opt)

        println("Minimum revenue = ", rev_min) 
        println("Minimum price = ", price_min)
    end
end