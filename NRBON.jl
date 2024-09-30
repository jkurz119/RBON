# Fit a normalized Radial Basis Network using projection operator training
# only normalized in terms of the input functions, not target function evaluation location input

function norm_fit!(rbon::RBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64})
    
    # h refers to higher network (input functions)
    # l refers to lower network (target function evaluation locations)
    # Perform k-means clustering on function input
    h_result = kmeans(U, rbon.h_num)  
    # Get cluster centers for higher network nodes
    rbon.h_centers = h_result.centers
    # cluster sigmas
    rbon.h_sigmas = [sum(std(U[:, h_result.assignments .== i], mean=rbon.h_centers[:,i], dims=2)) for i in 1:rbon.h_num]
    # Replace NaN sigma value resulting from single point cluster_id
    rbon.h_sigmas = replace(rbon.h_sigmas, NaN => 1.0)
    # Repeat, finding centers/sigmas for lower network

    # Use simple kmeans algorithm: kmeans_1d
    # y = reshape(y,1,:) <- Not necessary with how y comes in now 
    rbon.l_centers, l_assignments = kmeans_1d(y, rbon.l_num)
    rbon.l_sigmas = [sum(std(y[:, l_assignments .== i], mean=rbon.l_centers[:,i], dims=2)) for i in 1:rbon.l_num]
    rbon.l_sigmas = replace(rbon.l_sigmas, NaN => 1.0)


    all_weights = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(y, 2))
    num_input_funcs = size(U, 2)
    

    #Compute radial basis function for U and y input
    H = reduce(hcat,[(radial_basis_function(U, rbon.h_centers[:, i], rbon.h_sigmas[i])) for i in 1:rbon.h_num])'
    L = reduce(hcat,[(radial_basis_function(reshape(y, size(y, 1), :), rbon.l_centers[:, i], rbon.l_sigmas[i])) for i in 1:rbon.l_num])'
    
    #H_sum = sum(H, dims=1)

    # Product of hidden layers from upper networks with lower network
    Phi = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(U, 2))

    for j in 1:size(y,2)
        
        
        Phi =  reduce(hcat, [(H[:,k] * L[:,j]')[:] for k in 1:num_input_funcs]) # Flatten to vector of length column h_num * l_num
        
        ################# 
        # Least Squares
        #################
        weight_j = pinv(Phi)' * (V[j,:].* vec(sum(Phi, dims=1)))
        #weight_j = pinv(Phi)' * (V[j,:].*vec(H_sum))
        
        all_weights[:, j] = weight_j

    end

    # Calculate the row-wise average of all the weight vectors
    rbon.weights = mean(all_weights, dims=2)[:,1]

    #for scaling and shifting final output
    predictions = zeros(Float64, size(y,2), size(U,2))
    # Preallocate Phi2 matrix
    Phi2 = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(U, 2))

    for k in 1:size(U,2)
        for j in 1:size(y,2)
            # Compute the radial basis function for each test example
            #H = [radial_basis_function(reshape(U[:, j], :, 1), rbon.h_centers[:, i], rbon.h_sigmas[i]) for i in 1:rbon.h_num]
            #L = [radial_basis_function(reshape(y[:, k],:,1), rbon.l_centers[:, i], rbon.l_sigmas[i]) for i in 1:rbon.l_num]
            
            # Product of hidden layers from upper network with lower network
            Phi2 =  H[:, k] * L[:,j]'
            Phi2 = reduce(vcat, Phi2[:])
            
            # Compute the output and store it in the predictions matrix
            # Predictions matrix formatted [v_1, v_2,...,v_n] as columns for
            # output v_i corresponding to function input u_i
            # rows are evaluation of v_i at point y_k
            predictions[j, k] = (Phi2' * rbon.weights)[1,1] /sum(Phi2)
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
function norm_predict(rbon::RBON, U::Matrix{Float64}, y::Matrix{Float64})
    num_input_funcs = size(U, 2)
    num_target_pts = size(y, 2)
    
    # Initialize an empty matrix to store the results
    predictions = zeros(Float64, num_target_pts, num_input_funcs)
    
    H = reduce(hcat,[(radial_basis_function(U, rbon.h_centers[:, i], rbon.h_sigmas[i])) for i in 1:rbon.h_num])'
    L = reduce(hcat,[(radial_basis_function(reshape(y, size(y, 1), :), rbon.l_centers[:, i], rbon.l_sigmas[i])) for i in 1:rbon.l_num])'
    
    Phi = Matrix{Float64}(undef, rbon.l_num * rbon.h_num, size(U, 2))
    
    for j in 1:num_input_funcs
        for k in 1:num_target_pts
           
            # Product of hidden layers from upper network with lower network
            Phi = H[:, j] * L[:,k]'
            Phi = reduce(vcat, Phi[:])
            
            # Compute the output and store it in the predictions matrix
            # Predictions matrix formatted [v_1, v_2,...,v_n] as columns for
            # output v_i corresponding to function input u_i
            # rows are evaluation of v_i at point y_k
            predictions[k, j] = (Phi' * rbon.weights)[1,1] /sum(Phi)
        end
    end
    predictions = rbon.A * predictions .+ rbon.b
    return predictions
end