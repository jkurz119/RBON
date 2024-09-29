using MAT

##########################################################################
#
#           Wave Problem
#
##########################################################################

# data file located at https://drive.google.com/drive/folders/12cbNIvTp78YhXtl5j6Pp2KtxBzwdJCp5?usp=sharing
# Load the .mat file for Wave equation data, mapping wave initial condition to solution
reader = matread("Wave/Wave_data.mat")

# Read the fields
U = reader["U"] # matrix of initial condition functions (input)
V = reader["V"] # matrix of solutions
y = reader["y"] # domain location of solutions

using Random
Random.seed!(256)

U_train, U_val, U_test, V_train, V_val, V_test = train_val_test_split(U, V, 0.6, 0.14)

min_rel_norm = Inf
best_i, best_j = 0, 0

for i in 12:15
    for j in 2:15
        Random.seed!(1)
        # Create and fit the RBON
        temp_rbon = RBON(i, size(U_train, 1), j)
        rbon_fit!(temp_rbon, U_train, V_train, y)
        
        # Predict
        predictions = rbon_predict(temp_rbon, U_val, y)
        
        # Compute the relative norm
        rel_norm = norm(V_val .- predictions) / norm(V_val)
        
        # Check if this is the best so far
        if rel_norm < min_rel_norm
            min_rel_norm = rel_norm
            best_i, best_j = i, j
            printstyled("new min val error = $min_rel_norm, ", color=:blue)
            printstyled("new best i = $best_i, ", color=:green)
            printstyled("new best j = $best_j, ", color=:red)
        end
        print("Network widths tested this iteration: $i, $j\n ")

    end
end

# i,j = 4,11
Random.seed!(1)
rbon = RBON(4, size(U_train, 1), 11)

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
# i,j = 4,11
Random.seed!(1)
nrbon = RBON(4, size(U_train, 1), 11)
norm_fit!(nrbon, U_train, V_train, y)

#L2 relative test error, in distribution
n_predictions = norm_predict(nrbon, U_test, y)
n_rel_errors = [norm(V_test[:, i] .- n_predictions[:, i]) / norm(V_test[:, i]) for i in 1:size(V_test, 2)]
n_avg_error = mean(n_rel_errors)
standard_error = std(n_rel_errors) / sqrt(length(n_rel_errors))
n_MOE = z_value * standard_error


 
# OOD predictions and errors
reader = matread("Wave/Wave_data_ood.mat")

# Read the fields
U_ood = reader["U_ood"] # matrix of constant wave speeds (input)
V_ood = reader["V_ood"] # matrix of solutions
y_ood = reader["y_ood"] # domain location of solutions

# L2 relative test error, out of distribution for rbon
ood_predictions = rbon_predict(rbon, U_ood, y_ood)
ood_errors = [norm(V_ood[:, i] .- ood_predictions[:, i]) / norm(V_ood[:, i]) for i in 1:size(V_ood, 2)]

av_ood_error = mean(ood_errors)
ood_MOE = z_value * std(ood_errors) / sqrt(length(ood_errors))

# L2 relative test error, out of distribution for nrbon
n_ood_predictions = predict(nrbon, U_ood, y_ood)
n_ood_errors = [norm(V_ood[:, i] .- n_ood_predictions[:, i]) / norm(V_ood[:, i]) for i in 1:size(V_ood, 2)] 


av_n_ood_error = mean(n_ood_errors)
n_ood_MOE = z_value * std(n_ood_errors) / sqrt(length(n_ood_errors))


###############################################
#
#         Fourier "Grid Points"
#
###############################################
using FFTW


L = 1.5*pi
T = 240
n_t = 241 
n_x = 101

ts = range(0, T; length=n_t)
xs = range(0, L; length=n_x)

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
    Vfft_train[:,i] = fft(reshape(V_train[:,i],101,241))[:]
end

for i in 1:size(U_val,2)
    Ufft_val[:,i] = fft(U_val[:,i])
    Vfft_val[:,i] = fft(reshape(V_val[:,i],101,241))[:]
end

for i in 1:size(U_test,2)
    Ufft_test[:,i] = fft(U_test[:,i])
    Vfft_test[:,i] = fft(reshape(V_test[:,i],101,241))[:]
end

for i in 1:size(U_ood,2)
    Ufft_ood[:,i] = fft(U_ood[:,i])
    Vfft_ood[:,i] = fft(reshape(V_ood[:,i],101,241))[:]
end

# F-RBON ##########################################################
# 8,4
Random.seed!(1)
f_rbon = ComplexRBON(8, size(Ufft_train,1), 4)

complex_fit!(f_rbon, Ufft_train, Vfft_train, f_y)

comp_predictions = complex_predict(f_rbon, Ufft_test, f_y)

f_rel_errors = [norm(Vfft_test[:, i] .- comp_predictions[:, i]) / norm(Vfft_test[:, i]) for i in 1:size(Vfft_test, 2)]

average_error = mean(f_rel_errors)

using Statistics

standard_error = std(f_rel_errors) / sqrt(length(f_rel_errors))
z_value = 1.96  # for 95% confidence level
margin_of_error = z_value * standard_error

f_ood_predictions = complex_predict(f_rbon, Ufft_ood, f_y)

f_ood_errors = [norm(Vfft_ood[:, i] .- f_ood_predictions[:, i]) / norm(Vfft_ood[:, i]) for i in 1:size(Vfft_ood,2)]

average_ood_error = mean(f_ood_errors)

f_ood_MOE = z_value * std(f_ood_errors) / sqrt(length(f_ood_errors))
