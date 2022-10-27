using Dates, Plots, LaTeXStrings

function forward_euler(f, y_init, t_array)
    y_approx = Array{Float64, 1}(undef, length(t_array))
    y_approx[1] = y_init
    for i in 1:length(t_array)-1
        Δt = t_array[i+1] - t_array[i]
        y_approx[i+1] = y_approx[i] + Δt*f(t_array[i], y_approx[i])
    end
    return y_approx
end

function plot_residual()
    α = 0.4
    t_end = 1.0
    N = 5
    t_array = range(0, t_end, N+1) |> collect
    f(t, y) = (y - 2t*y^2) / (1 + t)
    y(t; A=α) = (1 + t)/(t^2 + 1/A)

    y_approx = forward_euler(f, α, t_array) # Incorrect approximation?
    my_plot = plot(
        t_array, y_approx,
        size=(1200, 900), thickness_scaling=2.0,
        ylimits=(0.3, 0.7), xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        xlabel=L"t", ylabel=L"y", markershape=:circle, markercolor=:green,
        markersize=2, markerstrokewidth=0.1, linecolor=:green,
        label="Approximate Solution", key=:bottomright)

    t_plot = 0:0.01:t_end
    plot!(my_plot, t_plot, y.(t_plot), linecolor=:black, label="Exact Solution")
    y_init_values = 0.0:0.02:0.8 |> collect
    deleteat!(y_init_values, findall(x -> x==α, y_init_values))
    for y_init in y_init_values
        plot!(my_plot, t_plot, y.(t_plot; A=y_init), linestyle=:dash, linecolor=:grey, label="")
    end


    display(my_plot)
end

function y_approx_func_linear(t, t_array, y_approx)
    Δt = t_array[2] - t_array[1]
    index_cont = 1 + (t - t_array[1])/Δt # Add 1 to account for Julia indexing from 1
    i = convert(Int64, floor(index_cont))
    return (t - t_array[i])*(y_approx[i+1] - y_approx[i])/(t_array[i+1] - t_array[i]) +  y_approx[i]
end

function make_y_approx_func_polynomial()

end

function residual_integral(t, α, y_approx_func; N=100)
    τ_values = range(0, t, N+1) |> collect
    dτ = τ_values[2] - τ_values[1]
    return α + sum(f(τ, y_approx_func(τ))*dτ for τ in τ_values)
end