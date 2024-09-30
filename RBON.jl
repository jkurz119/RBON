#cd(@__DIR__)
#using Pkg.activate(".")


using Clustering
using Plots


# Radial Basis Operator Network Class
mutable struct RBON
    h_num::Int
    l_num::Int
    h_centers::Matrix{Float64}
    l_centers::Matrix{Float64}
    h_sigmas::Vector{Float64}
    l_sigmas::Vector{Float64}
    weights::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}


    
    # h denotes the higher network that takes function values as input
    # l denotes the lower network that takes the values where the output functions
    # are ultimately evaluated (domain location of output function values)
    function RBON(h_num_arg::Int, num_u_evals::Int, l_num_arg::Int)
        # Initialize fields using the constructor arguments
        l_centers_arg = zeros(l_num_arg,1)
        l_sigmas_arg = zeros(l_num_arg)
        h_centers_arg = zeros(num_u_evals, h_num_arg)
        h_sigmas_arg = zeros(h_num_arg)
        weights_arg = zeros(l_num_arg * h_num_arg)

        # random initialization of weights
        #Random.seed!(315)  # Set a seed for reproducibility
        #rbon.weights = randn(rbon.h_num*rbon.l_num)       
        new(h_num_arg, l_num_arg, h_centers_arg, l_centers_arg,  h_sigmas_arg, l_sigmas_arg, weights_arg)
    end
end



# Fit the Radial Basis Network using projection operator training
function rbon_fit!(rbon::RBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64})
    

    # Perform k-means clustering on function input
    h_result = kmeans(U, rbon.h_num)  
    # Get cluster centers for higher network nodes
    rbon.h_centers = h_result.centers

    # Get cluster sigmas
    rbon.h_sigmas = [sum(std(U[:, h_result.assignments .== i], mean=rbon.h_centers[:,i], dims=2)) for i in 1:rbon.h_num]
    # Replace NaN sigma value resulting from single point cluster_id
    rbon.h_sigmas = replace(rbon.h_sigmas, NaN => 1.0)
    # Repeat, finding centers/sigmas for lower network
    # Use simple kmeans_1d since kmeans() does not handle case of dim(y) > 1
    # expects y to be in shape [y_1 y_2 ... y_n] 
    # where col y_i is a single point where output function v is measured 
    rbon.l_centers, l_assignments = kmeans_1d(y, rbon.l_num)
    rbon.l_sigmas = [sum(std(y[:, l_assignments .== i], mean=rbon.l_centers[:,i], dims=2)) for i in 1:rbon.l_num]
    rbon.l_sigmas = replace(rbon.l_sigmas, NaN => 1.0)

    all_weights = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(y, 2))
    num_input_funcs = size(U, 2)

    #Compute radial basis function for U and y input
    H = reduce(hcat,[(radial_basis_function(U, rbon.h_centers[:, i], rbon.h_sigmas[i])) for i in 1:rbon.h_num])'
    L = reduce(hcat,[(radial_basis_function(reshape(y, size(y, 1), :), rbon.l_centers[:, i], rbon.l_sigmas[i])) for i in 1:rbon.l_num])'
    
    Phi = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(U, 2))

    Threads.@threads for j in 1:size(y,2)
        # Product of hidden layers from upper networks with lower network
        Phi =  reduce(hcat, [(H[:,k] * L[:,j]')[:] for k in 1:num_input_funcs]) # Flatten to vector of length column h_num * l_num
        
        ################# 
        # Least Squares
        #################
        weight_j = pinv(Phi)' * V[j,:]
        # Stable inversion using Q R decomposition
        #Q, R = qr(Phi * Phi')
        #weight_j = R \ (Q' * (Phi * V[j, :]))
        all_weights[:, j] = weight_j
    end

    # Calculate the row-wise average of all the weight vectors
    rbon.weights = mean(all_weights, dims=2)[:,1]

    #output of RBON layer on training data for final affine transformation layer (scale & shift)
    predictions = zeros(Float64, size(y,2), size(U,2))
    
    Phi2 = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(U, 2))

    for k in 1:size(U,2)
        Threads.@threads  for j in 1:size(y,2)          
            # Product of hidden layers from upper network with lower network
            Phi2 = H[:, k] * L[:,j]'
            Phi2 = reduce(vcat, Phi2[:])
            
            # Compute the output and store it in the predictions matrix
            # Predictions matrix formatted [v_1, v_2,...,v_n] as columns for
            # output v_i corresponding to function input u_i
            # rows are evaluation of v_i at point y_k
            predictions[j, k] = (Phi2' * rbon.weights)[1,1] 
        end
    end

    # Reshape the equation to solve for A and b for affine transformation
    # Form the augmented matrix [results; ones(1, n)] for intercept
    pred_aug = [predictions; ones(1, size(predictions,2))]

    # Solve for [A; b] using least squares
    AB = V * pinv(pred_aug)

    # Extract A and b from the solution
    rbon.A = AB[:, 1:size(predictions,1)]
    rbon.b = AB[:, size(predictions,1)+1]
end

# Predict with the Radial Basis Operator Network
function rbon_predict(rbon::RBON, U::Matrix{Float64}, y::Matrix{Float64})
    num_input_funcs = size(U, 2)
    num_target_pts = size(y, 2)

    # Compute the output and store it in the predictions matrix
    # Predictions matrix formatted [v_1, v_2,...,v_n] as columns for
    # output v_i corresponding to function input u_i
    # rows are evaluation of v_i at point y_k

    # Initialize an empty matrix to store the results
    predictions = zeros(Float64, num_target_pts, num_input_funcs)

    #Compute radial basis function for U and y input
    H = reduce(hcat,[(radial_basis_function(U, rbon.h_centers[:, i], rbon.h_sigmas[i])) for i in 1:rbon.h_num])'
    L = reduce(hcat,[(radial_basis_function(reshape(y, size(y, 1), :), rbon.l_centers[:, i], rbon.l_sigmas[i])) for i in 1:rbon.l_num])'
    
    Phi = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(U, 2))
    
    Threads.@threads for j in 1:num_input_funcs
        for k in 1:num_target_pts
            # Product of hidden layers from upper network with lower network
            Phi = H[:, j] * L[:,k]'
            Phi = reduce(vcat, Phi[:])
            
            #multiply by weights
            predictions[k, j] = (Phi' * rbon.weights)[1,1] 
        end
    end

    #final linear transformation
    predictions = rbon.A * predictions .+ rbon.b
    return predictions
end

# Example usage:
# Create a Radial Basis Network with 50 nodes for the higher network, 100 nodes for lower network
# knowing the number of evals for training functions used for initialization purposes, can be altered as needed
#rbon = RBON(50, size(U,1), 100)