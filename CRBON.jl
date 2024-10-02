
# Radial Basis Operator Network Class
mutable struct ComplexRBON
    h_num::Int
    l_num::Int
    h_centers::Matrix{ComplexF64}
    l_centers::Matrix{Float64}
    h_sigmas::Vector{Float64}
    l_sigmas::Vector{Float64}
    weights::Vector{ComplexF64}
    A::Matrix{ComplexF64}
    b::Vector{ComplexF64}

    # h denotes the higher network that takes function values as input
    # l denotes the lower network that takes the values where the transformed functions
    # are ultimately evaluated 
    function ComplexRBON(h_num_arg::Int, num_u_evals::Int, l_num_arg::Int)
        # Initialize fields using the constructor arguments
        l_centers_arg = zeros(l_num_arg,1)
        l_sigmas_arg = zeros(l_num_arg)
        h_centers_arg = zeros(num_u_evals, h_num_arg)
        h_sigmas_arg = zeros(h_num_arg)
        weights_arg = zeros(l_num_arg * h_num_arg)

        # random initialization of weights
        #Random.seed!(123)  # Set a seed for reproducibility
        #rbon.weights = randn(rbon.h_num*rbon.l_num)
        
        new(h_num_arg, l_num_arg, h_centers_arg, l_centers_arg,  h_sigmas_arg, l_sigmas_arg, weights_arg)
    end
end


#accepts 2D arrays of complex values for training and testing
#assumes the second index is to access flattened fourier functions (U[:,k] -> kth input function)
#can be altered to accept arbitrary size input for larger dimensions
function complex_fit!(rbon::ComplexRBON, U::Array{ComplexF64, 2}, V::Array{ComplexF64, 2}, y::Matrix{Float64})


    rbon.h_centers, h_assignments = kmeans_complex(U, rbon.h_num)
    rbon.h_sigmas = [maximum(complex_std(U[:, h_assignments .== i], rbon.h_centers[:,i], size(U,1))) for i in 1:rbon.h_num] 

    # Use simple kmeans algorithm
    rbon.l_centers, l_assignments = kmeans_1d(y, rbon.l_num)
    rbon.l_sigmas = [sum(std(y[:, l_assignments .== i], mean=rbon.l_centers[:,i], dims=2)) for i in 1:rbon.l_num]
    #must add a replace NaN later
    rbon.l_sigmas = replace(rbon.l_sigmas, NaN => 1.0)

    all_weights = Array{ComplexF64}(undef, rbon.l_num * rbon.h_num, size(y, 2))
    num_input_funcs = size(U, 2)

    #Compute radial basis function for U and y input
    H = reduce(hcat,[(complex_rbf(U, rbon.h_centers[ :, i], rbon.h_sigmas[i])) for i in 1:rbon.h_num])'
    L = reduce(hcat,[(radial_basis_function(reshape(y, size(y, 1), :), rbon.l_centers[:, i], rbon.l_sigmas[i])) for i in 1:rbon.l_num])'
    
    Phi = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(U, 2))

    Threads.@threads for j in 1:size(y,2)

        Phi =  reduce(hcat, [(H[:,k] * L[:,j]')[:] for k in 1:num_input_funcs]) # Flatten to vector of length column h_num * l_num
        
        ################# 
        # Least Squares
        #################
        # Calculate the weight vector for j and store it in all_weights
        weight_j = pinv(Phi)' * V[j,:]
        # Stable inversion using Q R decomposition
        #Q, R = qr(Phi * Phi')
        #weight_j = R \ (Q' * (Phi * V[j, :]))
        all_weights[:, j] = weight_j

    end

    # Calculate the row-wise average of all the weight vectors
    rbon.weights = mean(all_weights, dims=2)[:,1]

    #for scaling and shifting final output
    predictions = zeros(ComplexF64, size(y,2), size(U,2))
    
    Phi2 = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(U, 2))

    Threads.@threads for k in 1:size(U,2)
        for j in 1:size(y,2)
            
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
    # Form the augmented matrix [results; ones(1, n)]
    pred_aug = [predictions; ones(1, size(predictions,2))]

    # Solve for [A; b] using least squares
    AB = V * pinv(pred_aug)

    # Extract A and b from the solution
    rbon.A = AB[:, 1:size(predictions,1)]
    rbon.b = AB[:, size(predictions,1)+1]


end

# Predict with the Radial Basis Operator Network
# Predictions matrix formatted [v_1, v_2,...,v_n] as columns for
# output v_i corresponding to function input u_i
# rows are evaluation of v_i at point y_k
function complex_predict(rbon::ComplexRBON, U::Array{ComplexF64, 2}, y::Matrix{Float64})
    num_input_funcs = size(U, 2)
    num_target_pts = size(y, 2)
    
    # Initialize an empty matrix to store the results
    predictions = zeros(ComplexF64, num_target_pts, num_input_funcs)

    H = reduce(hcat,[(complex_rbf(U, rbon.h_centers[ :, i], rbon.h_sigmas[i])) for i in 1:rbon.h_num])'
    L = reduce(hcat,[(radial_basis_function(reshape(y, size(y, 1), :), rbon.l_centers[:, i], rbon.l_sigmas[i])) for i in 1:rbon.l_num])'
    
    Phi = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, num_input_funcs)
    
    
    Threads.@threads for k in 1:size(U,2)
        for j in 1:size(y,2)
            
            # Product of hidden layers from upper network with lower network
            Phi = H[:, k] * L[:,j]'
            Phi = reduce(vcat, Phi[:])
            
        
            predictions[j, k] = (Phi' * rbon.weights)[1,1] 
        end
    end
    predictions = rbon.A * predictions .+ rbon.b
    return predictions
end
