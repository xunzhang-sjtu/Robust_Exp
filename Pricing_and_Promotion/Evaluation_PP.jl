function compute_Profit(N,a,b,c,Y_given,x_given,P_dag)
    price = sum(Y_given .* P_dag,dims=2)[:,1]
    utilities = ones(N+1)
    utilities[1:N] = exp.(vec(a .+ b .* price + c .* x_given))

    prob = zeros(N)
    rev = zeros(N)
    for i in 1:N
        prob[i] = utilities[i] / sum(utilities)
        rev[i] = prob[i] * price[i]
    end
    # println("total profit = ",sum(rev))
    return price,sum(rev)
end