using QuadGK
using LinearAlgebra
using Base.Threads

######################################
#
#           Data generator
#
######################################


########## a*sin(π*x) as initial condition

# Function to compute a_0
function compute_a0(ν, a)
    integrand(x) = exp(-((2*π*ν)^-1) * a * (1 - cos(π * x)))
    a0, _ = quadgk(integrand, 0, 1)
    return a0
end

# Function to compute a_n
function compute_an(n, ν, a)
    integrand(x) = exp(-((2*π*ν)^-1) * a * (1 - cos(π * x))) * cos(n * π * x)
    an, _ = quadgk(integrand, 0, 1)
    return 2 * an
end

# Function to compute u(x, t)
function u(x, t, ν, a0, an, max_n = 64)
    numerator = 2*π*ν * sum(an[n] * exp(-π^2 * (n^2) * t) * n * sin(n * π * x) for n in 1:max_n)
    denominator = a0 + sum(an[n] * exp(-π^2 * (n^2) * t) * cos(n * π * x) for n in 1:max_n)
    
    return numerator / denominator
end

# Example usage
ν = 0.01  # example value for ν
x = 0.1 # example value for x
t = 0.1  # example value for t
#u_xt = u(x, t, ν, 128)
#println("u(x, t) = $u_xt")


# Define ranges for x and t
xs = 0.0001:0.01:0.5
ts = 0.1:0.01:0.9


#generating data for changing amplitude, a, in (0.1, 1.0)
a_vals = 0.1:0.01:5.0

#input data, depends only on x (initial conditions)
U = [a*sin(pi*x) for x in xs, a in a_vals]

V = Matrix{Float64}(undef, length(xs) * length(ts), length(a_vals))


# parallelize the computation
Threads.@threads for i in 1:length(a_vals)
    a0 = compute_a0(ν, a_vals[i])
    an = [compute_an(n, ν, a_vals[i]) for n in 1:128]
    
    results = [u(x, t, ν, a0, an, 128) for x in xs, t in ts]
    V[:, i] = results[:]
end

# meshgrid function in utils
TX = meshgrid(ts, xs)

tt = TX[1][:]
xx = TX[2][:]

y = zeros(2, length(ts)*length(xs))
y[1,:] = tt
y[2,:] = xx


using MAT
matwrite("Burgers_data_sin_init.mat", Dict("V" => V,"U" => U,"y" => y))



########## b*x*(1-x) as initial condition

# only need to alter a0 and an
b_vals = 3.5:0.01:4.5


# Function to compute a_0
function compute_a0_poly(ν, b)
    integrand(x) = exp(-((12*ν)^-1) * b * x^2 * (3 - 2*x))
    a0, _ = quadgk(integrand, 0, 1)
    return a0
end

# Function to compute a_n
function compute_an_poly(n, ν, b)
    integrand(x) = exp(-((12*ν)^-1) * b * x^2 * (3 - 2*x)) * cos(n * π * x)
    an, _ = quadgk(integrand, 0, 1)
    return 2 * an
end

#input data, depends only on x (initial conditions)
U_ood = [b*x*(1-x) for x in xs, b in b_vals]

V_ood = Matrix{Float64}(undef, length(xs) * length(ts), length(b_vals))


# parallelize the computation
Threads.@threads for i in 1:length(b_vals)
    a0 = compute_a0_poly(ν, b_vals[i])
    an = [compute_an_poly(n, ν, b_vals[i]) for n in 1:128]
    
    results = [u(x, t, ν, a0, an, 128) for x in xs, t in ts]
    V_ood[:, i] = results[:]
end

#V_ood functions evaluated at same (t,x) as V
y_ood = y

matwrite("Burgers_data_poly_init.mat", Dict("V_ood" => V_ood,"U_ood" => U_ood,"y_ood" => y_ood))