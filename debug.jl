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
    y[1:p,:] = rand(Normal(0,1), p, n)
    
    # Generate VAR timeseries
    for t in p+1:T
        # Construct lagged variables matrix X
        X = zeros(1, n*p)
        for i in 1:p
            X[1, (i-1)*n+1:i*n] = y[t-i, :]
        end
        
        # Generate residuals from a multivariate normal distribution
        e[t, :] = rand(MvNormal(zeros(n), σ²), 1)
        
        # Calculate next time period's y value
        y[t, :] = β * X + e[t, :]
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