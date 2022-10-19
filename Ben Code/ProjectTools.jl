module ProjectTools

using LinearAlgebra, Statistics

export error_calculate, compute_integration_matrix

function error_calculate(data_correct, data_simulated, p)
    err_local = abs.(data_correct - data_simulated)
    err_global = abs.([sum((data_correct - data_simulated)[1:i]) for i in 1:length(data_correct)])
    err_norm = norm(err_data, p)
    err_max  = maximum(err_data)
    return Dict(
        :local => err_local,
        :global => err_global,
        :norm => err_norm,
        :max => err_max
    )
end

function compute_integration_matrix(M; integral_resolution=10)
    [compute_integration_matrix_entry(M, m, i, integral_resolution)
     for m = 0:(M-1), i = 0:M]
end

function compute_integration_matrix_entry(M, m, i, integral_resolution)
    t_array = range(m, m+1, integral_resolution) |> collect
    dt = t_array[2] - t_array[1]
    sum(prod((t-k)/(i-k) for k = 0:M if k != i)*dt for t in t_array[1:end-1])
end

end