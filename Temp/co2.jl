#=
    Code used for predicting temp based on co2

    Atmospheric co2 data source:
        C. D. Keeling, S. C. Piper, R. B. Bacastow, M. Wahlen, T. P. Whorf, M. Heimann, and       							
        H. A. Meijer, Exchanges of atmospheric CO2 and 13CO2 with the terrestrial biosphere and   							
        oceans from 1978 to 2000.  I. Global aspects, SIO Reference Series, No. 01-06, Scripps    							
        Institution of Oceanography, San Diego, 88 pages, 2001.                                   							

    Average global temperature source:
        https://ourworldindata.org/grapher/monthly-average-surface-temperatures-by-year

=#



using Random
using CSV
using DataFrames
using Plots
# Read the CSV file into a DataFrame
df = CSV.read("Temp/MonthlyCo2_GlobalTemp.csv", DataFrame)

# Convert columns to arrays
date_array = df.Date
global_temp_array = df.Temp
co2_array = df.CO2

#reshape, each column corresponds the twelve
#measurements for that year
V = reshape(global_temp_array[1:768], 12, :)
U = reshape(co2_array[1:768], 12, :)
y = Float64.(1:12)

#train on older data
final_idx = 59
U_train =  U[:,1:final_idx]
V_train = V[:,1:final_idx]
y_train = y

# 5, 12 
# 27, 5 
rbon = RBON(5, size(U,1), 12)
rbon_norm = RBON(27, size(U,1), 5)

# Fit the network
Random.seed!(1) 
rbon_fit!(rbon, U_train, V_train, reshape(y_train,1,:))
Random.seed!(1) 
norm_fit!(rbon_norm, U_train, V_train, reshape(y_train,1,:))

# RBON redictions on next two years
U_test = U[:,60:64]
V_test = V[:,60:64]
predictions = rbon_predict(rbon, U_test, reshape(y_train,1,:))

#average L2 relative error
rel_errors = [norm(V_test[:, i] .- predictions[:, i]) / norm(V_test[:, i]) for i in 1:size(V_test, 2)]

average_error = mean(rel_errors)

#normRBON predictions
norm_pred = norm_predict(rbon_norm, U, reshape(y_train,1,:))
rel_norm_errors = [norm(V_test[:, i] .- norm_pred[:, i]) / norm(V_test[:, i]) for i in 1:size(V_test, 2)]
average_norm_error = mean(rel_norm_errors)

#predictions on all the data 
predictions_all = rbon_predict(rbon, U, reshape(y_train,1,:))
norm_pred_all = norm_predict(rbon_norm, U, reshape(y_train,1,:))

x_values = date_array[1:768]



using Plots

#start plot at 1970

plot(size=(1200, 800), dpi=300)


colors = [:blue, :lightcoral, :black] 


plot(x_values[121:end], predictions_all[:,11:end][:], 
     label="RBON", linestyle=:dot, linewidth=2, color=colors[1],
     xlabel="Year", ylabel="Temperature (°C)",
     legend=:topleft,  
     grid=false,  
     background_color=:white)  

plot!(x_values[121:end], norm_pred_all[:,11:end][:], 
      label="NRBON", linestyle=:dash, linewidth=2, color=colors[2])

plot!(x_values[121:end], V[:,11:end][:], 
      label="Avg Temp (°C)", linestyle=:solid, linewidth=1, color=colors[3])



savefig("temperature_predictions.pdf")


light_grey = RGB(0.9, 0.9, 1.0)  # Very light color


p = plot(x_values[661:end], predictions_all[:,56:end][:], 
     label="RBON", linestyle=:dot, linewidth=2, color=colors[1],
     xlabel="Year", ylabel="Temperature (°C)",
     legend=:topleft,  
     grid=false, 
     background_color=:white)  

plot!(p, x_values[661:end], norm_pred_all[:,56:end][:], 
      label="NRBON", linestyle=:dash, linewidth=2, color=colors[2])

plot!(p, x_values[661:end], V[:,56:end][:], 
      label="Avg Temp (°C)", linestyle=:solid, linewidth=1, color=colors[3])

# Add light blue background starting from x_values[745]
vspan!(p, [x_values[745], x_values[end]], fillcolor=light_blue, fillalpha=0.3, label="")



# Save the plot as a high-quality PDF
savefig(p, "temperature_pred.pdf")