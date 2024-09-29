
#=
    Code used for the Euler-Bernoulli beam equation in the RBON paper

    data.mat file is downloaded from
    https://github.com/qianyingcao/Laplace-Neural-Operator/blob/main/3D_Brusselator/main.py

=#

using MAT


# Load the .mat file
reader = matread("Beam/data.mat")

# Read the fields
U_pre1 = reader["f_train"]
V_pre1 = reader["u_train"]
ts = reader["t"]
xs = reader["x"]

U_pre2 = reader["f_vali"]
V_pre2 = reader["u_vali"]

U_p_ood = reader["f_test"]
V_p_ood = reader["u_test"]

U = zeros(size(U_pre1,2)*size(U_pre1,3), size(U_pre1,1) + size(V_pre2,1))
V = zeros(size(V_pre1,2)*size(V_pre1,3), size(V_pre1,1) + size(V_pre2,1))
U_ood = zeros(size(U_p_ood,2)*size(U_p_ood,3), size(U_p_ood,1))
V_ood = zeros(size(V_p_ood,2)*size(V_p_ood,3), size(V_p_ood,1))

# combine train and validation data to reshuffle and
# create a train, validation, in distribution test set
for i in 1:size(U_pre1, 1)
    U[:,i] = U_pre1[i,:,:][:] #flatten
    V[:,i] = V_pre1[i,:,:,][:]
end

for i in 1:size(U_pre2, 1)
    U[:,i+size(U_pre1, 1)] = U_pre2[i,:,:][:] #flatten
    V[:,i+size(U_pre1, 1)] = V_pre2[i,:,:,][:]
end

#flatten testing data
for i in 1:size(U_p_ood, 1)
    U_ood[:,i] = U_p_ood[i,:,:][:]
    V_ood[:,i] = V_p_ood[i,:,:,][:]
end

TX = meshgrid(ts, xs)

tt = TX[1][:]
xx = TX[2][:]
y = zeros(2, length(ts)*length(xs))
y[1,:] = tt
y[2,:] = xx

using Random
#rand() produces 0.07336635446929285 for Random.seed!(1)
#Julia versions older than 1.7 use a different random number generator
Random.seed!(1) 

U_train, U_val, U_test, V_train, V_val, V_test = train_val_test_split(U, V, 0.6, 0.14)

# 15, 4
Random.seed!(1)
rbon = RBON(15, size(U_train, 1), 4)

rbon_fit!(rbon, U_train, V_train, y)

# L2 relative test error, in distribution
predictions = rbon_predict(rbon, U_test, y)
rel_errors = [norm(V_test[:, i] .- predictions[:, i]) / norm(V_test[:, i]) for i in 1:size(V_test, 2)]

average_error = mean(rel_errors)

using Statistics

standard_error = std(rel_errors) / sqrt(length(rel_errors))
z_value = 1.96  # for 95% confidence level
margin_of_error = z_value * standard_error

### NRBON
# i,j = 15,4
Random.seed!(1)
nrbon = RBON(15, size(U_train, 1), 4)
norm_fit!(nrbon, U_train, V_train, y)

#L2 relative test error, in distribution
n_predictions = norm_predict(nrbon, U_test, y)
n_rel_errors = [norm(V_test[:, i] .- n_predictions[:, i]) / norm(V_test[:, i]) for i in 1:size(V_test, 2)]
n_avg_error = mean(n_rel_errors)
standard_error = std(n_rel_errors) / sqrt(length(n_rel_errors))
z_value = 1.96
n_MOE = z_value * standard_error


# L2 relative test error, out of distribution for nrbon
n_ood_predictions = norm_predict(nrbon, U_ood, y)
n_ood_errors = [norm(V_ood[:, i] .- n_ood_predictions[:, i]) / norm(V_ood[:, i]) for i in 1:size(V_ood,2)] 


av_n_ood_error = mean(n_ood_errors)
n_ood_MOE = z_value * std(n_ood_errors) / sqrt(length(n_ood_errors))



###############################################
#
#         Fourier Grid Points
#
###############################################
using FFTW

#convert vector to Range object
ts = ts[1]:ts[2]-ts[1]:ts[length(ts)]
xs = xs[1]:xs[2]-xs[1]:xs[length(xs)]

# Determine the Fourier grid points
n_t = length(ts)
n_x = length(xs)
k_x = fftshift(fftfreq(n_t, step(vec(ts))))
k_y = fftshift(fftfreq(n_x, step(xs)))

# Fourier grid points
TX = meshgrid(k_x, k_y)

k_xx = TX[1][:]
k_yy = TX[2][:]
f_y = zeros(2, length(k_x)*length(k_y))
f_y[1,:] = k_xx
f_y[2,:] = k_yy


Ufft_train = Array{ComplexF64}(undef, size(U_train,1), size(U_train, 2))
Vfft_train = Array{ComplexF64}(undef, size(V_train,1), size(V_train, 2))

Ufft_val = Array{ComplexF64}(undef, size(U_val, 1), size(U_val, 2))
Vfft_val = Array{ComplexF64}(undef, size(V_val,1), size(V_val, 2))

Ufft_test = Array{ComplexF64}(undef, size(U_test, 1), size(U_test, 2))
Vfft_test = Array{ComplexF64}(undef, size(V_test,1), size(V_test, 2))

Ufft_ood = Array{ComplexF64}(undef, size(U_ood, 1), size(U_ood, 2))
Vfft_ood = Array{ComplexF64}(undef, size(V_ood,1), size(V_ood, 2))

for i in 1:size(U_train,2)
    Ufft_train[:,i] = fft(U_train[:,i])
    Vfft_train[:,i] = fft(reshape(V_train[:,i],n_x,n_t))[:]
end

for i in 1:size(U_val,2)
    Ufft_val[:,i] = fft(U_val[:,i])
    Vfft_val[:,i] = fft(reshape(V_val[:,i],n_x,n_t))[:]
end

for i in 1:size(U_test,2)
    Ufft_test[:,i] = fft(U_test[:,i])
    Vfft_test[:,i] = fft(reshape(V_test[:,i],n_x,n_t))[:]
end

for i in 1:size(U_ood,2)
    Ufft_ood[:,i] = fft(U_ood[:,i])
    Vfft_ood[:,i] = fft(reshape(V_ood[:,i],n_x,n_t))[:]
end


# F-RBON ######################################################## 
# network widths 8, 8
Random.seed!(1)
f_rbon = ComplexRBON(8, size(Ufft_train,1), 8)

complex_fit!(f_rbon, Ufft_train, Vfft_train, f_y)

comp_predictions = complex_predict(f_rbon, Ufft_test, f_y)

f_rel_errors = [norm(Vfft_test[:, i] .- comp_predictions[:, i]) / norm(Vfft_test[:, i]) for i in 1:size(Vfft_test, 2)]

average_error = mean(f_rel_errors)

standard_error = std(f_rel_errors) / sqrt(length(f_rel_errors))
z_value = 1.96  # for 95% confidence level
margin_of_error = z_value * standard_error

f_ood_predictions = complex_predict(f_rbon, Ufft_ood, f_y)

f_ood_errors = [norm(Vfft_ood[:, i] .- f_ood_predictions[:, i]) / norm(Vfft_ood[:, i]) for i in 1:size(Vfft_ood,2)]

average_ood_error = mean(f_ood_errors)
z_value = 1.96
f_ood_MOE = z_value * std(f_ood_errors) / sqrt(length(f_ood_errors))


