function Generate_Params(N,is_original)
    if is_original
        if abs(N-3) >= 0.0001
            println("Warning: N is not 3")
        else
            a = [1.0,1.0,1.0]
            b = [-0.1,-0.2,-0.3]
            c = [0.8,0.3,0.5]
        end
    else
        a = ones(N)
        b = round.(collect(range(0.1, 0.1*N, length=N)),digits=2)
        c = round.(rand(Uniform(0.0, 1.0), N),digits=2)
    end
    return a,b,c
end

function Generate_Data(a,b,c,N, N_u, S,P_bar)
    P_sample = round.(rand(S,N) .* P_bar,digits=2);

    indices = rand(1:N, S)
    PM_sample = zeros(Int, S, N)
    PM_sample[CartesianIndex.(1:S, indices)] .= 1
    choice_sample = zeros(S)
    for s in 1:S
        price = P_sample[s,:]
        PP = PM_sample[s,:]
        utilities = ones(N+1) # the N+1 is the no purchase choice
        utilities[1:N] = exp.(vec(a .+ b .* price + c .* PP))
        prob = utilities ./sum(utilities)
        choice_sample[s] = sample(1:(N+1), Weights(prob))
    end

    return P_sample, PM_sample, choice_sample
end