using Statistics, LinearAlgebra, Distributions

#simple test to make sure that covariance in julia is consistent with definition given in the paper
a = [1, 2, 3, 4, 5, 8, 8, 8, 8, 8]

b = [1, 6, 7, 2, 3, 9, 7, 6, 3, 4]

cov(a,b)

#it is confirmed that the cov is the right definition

#now find the cofactors

function cofactor(C, x, y)
minor = det(C[1:end .!= x, 1:end .!= y])
return (-1)^(x + y)*minor
end

#cofactor test
a[1:end .!= 2, 1:end .!= 2]

#compute euler difference series
function Euler_diff(x,k,t=1)
    diff = circshift(x,-k) - x
    return diff[1:end-k]/(k*t)
end

#euler difference test
Euler_diff(a,1)

C = cov(reshape([a;b], 10, 2))

det(C)

#get information transfer value
function T(X, i, j, k=1, t=1)
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

iterations = 500

tseries = zeros(6,iterations)
tseries[:,1] = e=rand(Normal(), 6)
for i in 1:iterations-1
    println(i)
    tseries[:,i+1] = next(tseries[:,i])
end

d = size(tseries,1)
Transfer = zeros(d,d)
for i in 1:d
    for j in 1:d
        a = abs.(T(tseries',i,j))
        if a > 0.009
        Transfer[i,j] = T(tseries',i,j)
        else
        Transfer[i,j]=0
        end
    end
end

abs.(Transfer')
