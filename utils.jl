using LinearAlgebra
using StatsBase

# Radial Basis Function
function radial_basis_function(x::Matrix{Float64}, center::Vector{Float64}, sigma::Float64)
    # Initialize an empty vector to store the results
    num_columns = size(x, 2)
    output = zeros(Float64, num_columns)

    for j in 1:num_columns
        # Calculate the radial basis function for each column of x
        output[j] = exp(-norm(x[:, j] - center)^2 / (2 * sigma^2))
    end
    return output
end

#radial basis function for complex input
function complex_rbf(x::Array{ComplexF64, 2}, center::Array{ComplexF64, 1}, sigma::Float64)
    # Initialize an empty vector to store the results
    num_functions = size(x, 2)
    output = zeros(Float64, num_functions)

    for j in 1:num_functions
        # Calculate the radial basis function for each function in x
        output[j] = exp(-norm(x[ :, j] - center)^2 / (length(x[:, j]) * sigma^2))
        #output[j] = exp(-norm(x[ :, j] - center)^2 / (2 * sigma^2))
    end

    return output
end

# simple kmeans clustering function
# works for larger dimension but needed for 1D case since kmeans in Cluster.jl requires d>1
# put in as row vector
function kmeans_1d(X::Matrix{Float64}, num_clusters::Int, max_iter = 100, threshold = 0.001)
    k = num_clusters
    # Let's pick k points from X without replacment
    centroids = X[:, StatsBase.sample(1:size(X,2), k, replace = false)]

    # create a copy to check if the centroids are moving or not.
    new_centroids = copy(centroids)

    # start an empty array for our cluster ids. This will hold the cluster assignment
    # for each point in X
    cluster_ids = zeros(Float64, size(X, 2))

    for _ in 1:max_iter
        for col_idx in 1:size(X, 2)

            p = X[:, col_idx]

            # calculate the distance between the point and each centroid
            point_difference = mapslices(x -> x .- p, centroids, dims=[1])

            # we calculate the squared Euclidian distance
            distances = mapslices(sum, point_difference .^ 2, dims=[1])

            # now find the index of the closest centroid
            cluster_ids[col_idx] = findmin(distances)[2][2]
        end

        # iterate over each centroid
        for cluster_id in 1:size(centroids, 2)

            # find the mean of the assigned points for that particluar cluster
            mask = [i for (i, m) in enumerate(cluster_id .== cluster_ids) if m]
            new_centroids[:, cluster_id] = mapslices(mean, X[:, mask], dims=[2])
        end

        # total distance that the centroids moved
        center_change = sum(mapslices(x -> sum(x.^2), new_centroids .- centroids, dims=[2]))

        centroids = copy(new_centroids)

        # if the centroids move negligably, then we're done
        if center_change < threshold
            break
        end
    end

    return centroids, cluster_ids
 end

 #kmeans for complex input
 function kmeans_complex(X::Matrix{ComplexF64}, num_clusters::Int, max_iter = 100, threshold = 0.001)
    k = num_clusters
    # Let's pick k points from X without replacment
    centroids = X[:, StatsBase.sample(1:size(X,2), k, replace = false)]

    # create a copy. This is used to check if the centroids are moving or not.
    new_centroids = copy(centroids)

    # start an empty array for our cluster ids. This will hold the cluster assignment
    # for each point in X
    cluster_ids = zeros(Float64, size(X, 2))

    for _ in 1:max_iter
        for col_idx in 1:size(X, 2) 

            p = X[:, col_idx]

            #complex difference
            point_difference = mapslices(x -> x .- p, centroids, dims=[1])

           # Calculate the squared magnitude of the complex difference
            squared_magnitude = mapslices(x -> real(x .* conj(x)), point_difference, dims=[1])

            distances = mapslices(sum, squared_magnitude, dims=[1])

            # now find the index of the closest centroid
            cluster_ids[col_idx] = findmin(distances)[2][2]
        end

        # Iterate over each centroid
        for cluster_id in 1:size(centroids, 2)

            # find the mean of the assigned points for that particluar cluster
            mask = [i for (i, m) in enumerate(cluster_id .== cluster_ids) if m]
            new_centroids[:, cluster_id] = mapslices(mean, X[:, mask], dims=[2])
        end

        # now measure the total distance that the centroids moved
        center_change = sum(mapslices(x -> sum(real(x .* conj(x))), new_centroids .- centroids, dims=[2]))

        centroids = copy(new_centroids)

        # break if the centroids move negligably
        if center_change < threshold
            break
        end
    end

    return centroids, cluster_ids
 end

 # method for calculating the spread of the gaussians when using complex input
 function complex_std(X::Matrix{ComplexF64}, c::Vector{ComplexF64}, n::Int)

    #take conjugate difference for magnitude
    conjugate_diff = real((X .- c) .* conj(X .- c))

    #average the magnitudes in each column (across rows)
    averages = sum(conjugate_diff)./n

    sigma = sqrt.(averages)

    return sigma
end

# Function to split the matrix into training and test sets, assuming U and V have the same col_num
function train_test_split(U, V, train_ratio=0.8)
    n = size(U, 2)  # Number of columns
    indices = collect(1:n)  # Column indices
    shuffle!(indices)  # Shuffle the indices

    train_size = round(Int, train_ratio * n)
    train_indices = indices[1:train_size]
    test_indices = indices[train_size+1:end]

    U_train = U[:, train_indices]
    V_train = V[:, train_indices]
    U_test = U[:, test_indices]
    V_test = V[:, test_indices]

    return U_train, U_test, V_train, V_test
end

function train_val_test_split(U, V, train_ratio=0.6, val_ratio=0.2)
    n = size(U, 2)  # Number of columns
    indices = collect(1:n)  # Column indices
    shuffle!(indices)  # Shuffle the indices

    train_size = round(Int, train_ratio * n)
    val_size = round(Int, val_ratio * n)

    train_indices = indices[1:train_size]
    val_indices = indices[train_size+1:train_size+val_size]
    test_indices = indices[train_size+val_size+1:end]

    U_train = U[:, train_indices]
    V_train = V[:, train_indices]
    U_val = U[:, val_indices]
    V_val = V[:, val_indices]
    U_test = U[:, test_indices]
    V_test = V[:, test_indices]

    return U_train, U_val, U_test, V_train, V_val, V_test
end


function meshgrid(xin,yin)
    nx=length(xin)
    ny=length(yin)
    xout=zeros(ny,nx)
    yout=zeros(ny,nx)
    for jx=1:nx
        for ix=1:ny
            xout[ix,jx]=xin[jx]
            yout[ix,jx]=yin[ix]
        end
    end
    return (x=xout, y=yout)
end