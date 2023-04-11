#Untested so far,

using Statistics, Distributions
using StatsModels, GLM

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


#kuramoto model (Need to add matrix A to dtheta to create a causal network)

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


#VAR model

using Distributions

function generate_var_data(n::Int, T::Int, p::Int, β::Array{A,2}, σ²::Array{A,2}) where A <: Real
    # n is the number of variables
    # T is the number of time periods
    # p is the order of the VAR model
    # β is a n*p x n matrix of VAR parameters
    # σ² is a n x n covariance matrix of residuals
    
    # Set up variables
    y = zeros(T, n)
    e = zeros(T, n)
    
    # Generate initial values from a normal distribution
    y[1,:] = rand(Normal(0,1), n)
    
    # Generate VAR timeseries
    for t in 2:T
        # Construct lagged variables matrix X
        X = zeros(1, n*p)
        for i in 1:p
            X[1, (i-1)*n+1:i*n] = y[t-i, :]
        end
        
        # Generate residuals from a multivariate normal distribution
        e[t, :] = rand(MvNormal(zeros(n), σ²), 1)
        
        # Calculate next time period's y value
        y[t, :] = X * β + e[t, :]
    end
    
    return y, e
end

# Example usage
n = 3  # number of variables
T = 100  # number of time periods
p = 2  # VAR model order
β = [0.5 0.3 -0.2; -0.1 0.7 0.1; 0.2 -0.5 0.6]  # VAR parameters
σ² = [1.0 0.5 0.3; 0.5 1.0 0.2; 0.3 0.2 1.0]  # residuals covariance matrix
y, e = generate_var_data(n, T, p, β, σ²)
