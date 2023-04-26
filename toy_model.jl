using Statistics, Distributions
using StatsModels, GLM, LinearAlgebra, CSV, Tables

## Granger function is currently not working

# Define a function for computing the Granger causality from x to y
function grangercausality(x, y, p)
    # Combine x and y into a matrix
    data = hcat(x[p+1:end], y[p:end-1])

    # Split the data into a training set and a test set
    n_train = Int(floor(0.8 * size(data, 1)))
    train = data[1:n_train, :]
    test = data[n_train+1:end, :]

    # Fit two autoregressive models using the training data
    model1 = StatsModels.fit(LinearModel, train[:, 1:p], train[:, 1])
    model2 = StatsModels.fit(LinearModel, train, train[:, 2])

    # Compute the prediction error using the test data
    y_pred1 = StatsModels.predict(model1, test[:, 1:p])
    y_pred2 = StatsModels.predict(model2, test)
    err1 = test[:, 1] - y_pred1
    err2 = test[:, 2] - y_pred2

    # Compute the Granger causality measure and its p-value
    rss1 = sum(err1.^2)
    rss2 = sum(err2.^2)
    F = ((rss1 - rss2) / p) / (rss2 / (n_train - 2p - 1))
    pval = ccdf(FDist(p, n_train - 2p - 1), F)

    return F, pval
end


# Generate some example data
using Random
Random.seed!(124)
x = randn(1000)
y = randn(1000)

# Compute the Granger causality from x to y
gc_xy = grangercausality(x, y, 1)

# Compute the Granger causality from y to x
gc_yx = grangercausality(y, x, 1)

# Print the results
println("Granger causality from x to y: $(gc_xy[1]) (p-value: $(gc_xy[2]))")
println("Granger causality from y to x: $(gc_yx[1]) (p-value: $(gc_yx[2]))")


#kuramoto model (Need to add matrix A to dtheta to create a causal network) (Not tested or working)

# Set the parameters of the Kuramoto model
const N = 100 # Number of oscillators
const K = 0.5 # Coupling strength
const w0 = 1.0 # Natural frequency

# Set the initial conditions
theta = 2π*rand(N)
omega = w0 .+ 0.1*w0*randn(N)

# Set the simulation parameters
const dt = 0.01 # Time step
const tmax = 100 # Maximum simulation time
t = 0:dt:tmax

# Define the Kuramoto function
function kuramoto(theta, omega, K)
    dtheta = copy(omega)
    for i in 1:N
        dtheta[i] += K/N * sum(sin.(theta .- theta[i]))
    end
    return dtheta
end

thetas = [theta]
for i in 2:length(t)
    push!(thetas, thetas[end] + kuramoto(thetas[end], omega, K) * dt)
end




#VAR model --------------------------------------------------------------------

function generate_var_data(n::Int, T::Int, p::Int, β::Array{A,3}, σ²::Array{A,2}) where A <: Real
    # n is the number of variables
    # T is the number of time periods
    # p is the order of the VAR model
    # β is a n*p x n matrix of VAR parameters
    # σ² is a n x n covariance matrix of residuals
    
    # Set up variables
    y = zeros(T, n)
    e = zeros(T, n)
    
    # Generate initial values from a normal distribution
    #y[1:p,:] = rand(Normal(0,1), p, n)
    y[1:p,:] = ones(p,n)

    
    # Generate VAR timeseries
    for t in p+1:T
        # Construct lagged variables matrix X
        X = zeros(p, n)
        for i in 1:p
            X[i,:] = y[t-i, :]
        end
        
        # Generate residuals from a multivariate normal distribution
        e[t, :] = rand(MvNormal(zeros(n), σ²), 1)
        
        # Calculate next time period's y value
        y[t, :] = reduce(+, [(X[i,:]')*(β[:,:,i]) for i in 1:size(β,3)]) + e[t, :]'
    end
    
    return y, e
end

#convert an a higher order var model to that of order 1
function convert_order1(C)
    # create a sample 3D array A

    # permute the dimensions of A to stack the 2D arrays vertically
    C_permuted = permutedims(C, (1, 3, 2))
    
    # reshape A_permuted to a 2D array B
    C_reshape = reshape(C_permuted, (size(C_permuted, 1)*size(C_permuted, 2), size(C_permuted, 3)))

    #build the auxillary shape
    B = zeros(size(C_reshape,1), 2*size(C_reshape,2))
    width = 2*size(C_reshape,2)
    B[1:width, 1:width] = I(width)

    #combine the auxilary array
    return hcat(C_reshape, B)

end

#convert an a higher order var model to that of order 1
function convert_order1(C)
    # create a sample 3D array A

    # permute the dimensions of A to stack the 2D arrays vertically
    C_permuted = permutedims(C, (1, 3, 2))
    
    # reshape A_permuted to a 2D array B
    C_reshape = reshape(C_permuted, (size(C_permuted, 1)*size(C_permuted, 2), size(C_permuted, 3)))

    #build the auxillary shape
    B = zeros(size(C_reshape,1), 2*size(C_reshape,2))
    width = 2*size(C_reshape,2)
    B[1:width, 1:width] = I(width)

    #combine the auxilary array
    return hcat(C_reshape, B)

end

r= sqrt(2)

# Example usage
n = 5  # number of variables
T = 10000  # number of time periods
p = 3  # VAR model order
A = zeros(n,n,p)
σ² = Matrix{Float64}(I, n, n)  # residuals covariance matrix


A[1,1,1] =  0.95*r;
A[1,1,2] = -0.9025;
A[2,1,2] =  0.5;
A[3,1,3] = -0.4;
A[4,1,2] = -0.5;
A[4,4,1] =  0.25*r;
A[4,5,1] =  0.25*r;
A[5,4,1] = -0.25*r;
A[5,5,1] =  0.25*r;


#=
A[1,3,1] = -0.6
A[2,1,1] = -0.5
A[2,6,1] = 0.8
A[3,2,1] = 0.7
A[4,4,1] = 0.7
A[4,5,1] = 0.4
A[5,4,1] = 0.2
A[5,6,1] = 0.7
A[6,6,1] = -0.5
=#


A = convert_order1(A)
# Example usage
n = 5  # number of variables
T = 10000  # number of time periods
p = 1  # VAR model order

σ² = 0.000000001 .* I(3*n)
σ²[1:5,1:5] = Matrix{Float64}(I, n, n)  # residuals covariance matrix
σ² = convert(Matrix{Float64}, σ²)
n=15


y, e = generate_var_data(n, T, p, A, σ²)

CSV.write("liangvartest.csv",  Tables.table(y), writeheader=false)
