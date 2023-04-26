using Statistics, LinearAlgebra, Distributions

#find the cofactors
function cofactor(C, x, y)
minor = det(C[1:end .!= x, 1:end .!= y])
return (-1)^(x + y)*minor
end

#compute euler difference series
function Euler_diff(x,k,t=1)
    diff = circshift(x,-k) - x
    return diff[1:end-k]/(k*t)
end

#get information transfer value
function T_info(X, i, j, k=1, t=1)
    C = cov(X)
    C_ratio = C[i,j]/C[i,i]
    di = Euler_diff(X[:,i], k, t)
    total = 0
    for l in 1:size(X)[2]
        cd = cov(X[1:end-k,l], di)
        delta = (cofactor(C, j, l))
        total += cd*delta
    end
    T = total*C_ratio / det(C)
    return T
end

#note, input formatat is an arary (each row is a time series)
#returns a matrix for each pariwise information measure
function T_causality(tseries, limit)
    d = size(tseries,1)
    Transfer = zeros(d,d)
    for i in 1:d
        for j in 1:d
            a = abs.(T_info(tseries',i,j))
            if a > limit
            Transfer[i,j] = T_info(tseries',i,j)
            else
            Transfer[i,j]=0
            end
        end
    end

    return Transfer
end





## Below is a test implementation of the above

#test system according to the paper
A = [0 0 -0.6 0 0 0; -0.5 0 0 0 0 0.8; 0 0.7 0 0 0 0; 0 0 0 0.7 0.4 0; 0 0 0 0.2 0 0.7; 0 0 0 0 0 -0.5]
B = diagm(ones(6))

function next(x_n)
    A = [0 0 -0.6 0 0 0; -0.5 0 0 0 0 0.8; 0 0.7 0 0 0 0; 0 0 0 0.7 0.4 0; 0 0 0 0.2 0 0.7; 0 0 0 0 0 -0.5]
    B = diagm(100*ones(6))
    alpha = [0.1; 0.7; 0.5; 0.2; 0.8; 0.3]
    e=rand(Normal(), 6)
    return alpha + A*x_n + B*e
end

iterations = 10000

tseries = zeros(6,iterations)
tseries[:,1] = e=rand(Normal(), 6)
for i in 1:iterations-1
    println(i)
    tseries[:,i+1] = next(tseries[:,i])
end

transfer = T_causality(tseries, 0.0009)

abs.(Transfer')[1:5,1:5]

#[x == 0.0 ? 0.0 : round(x/abs(x)) for x in Transfer]

#figure = heatmap(1:size(final,1), 1:size(final,2), final - (I(5)'), c=cgrad([:white, :black]))