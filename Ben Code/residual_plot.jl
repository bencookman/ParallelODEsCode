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
    t_end = 3.0
    N = 8
    f(t, y) = (y - 2t*y^2) / (1 + t)
    y(t; A=α) = (1 + t)/(t^2 + 1/A)
    # α = 0.0
    # t_end = pi
    # N = 20
    # f(t, y) = cos(t)
    # y(t; A=α) = sin(t) + A

    t_array = range(0, t_end, N+1) |> collect
    y_approx = forward_euler(f, α, t_array)
    residual_plot = plot(
        t_array, y_approx,
        size=(1200, 900), thickness_scaling=2.0,
        ylimits=(0.2, 0.8),
        xlabel=L"t", ylabel=L"y", markershape=:circle, markercolor=:green,
        markersize=2, markerstrokewidth=0.1, linecolor=:green,
        label="Approximate Solution", key=:bottomright
    )

    t_plot = 0:0.01:t_end |> collect
    plot!(residual_plot, t_plot, y.(t_plot), linecolor=:black, label="Exact Solution")
    y_init_values = 0.001:0.2:4.0 |> collect |> (x ->  1 ./ x)
    println(y_init_values)
    deleteat!(y_init_values, findall(x -> x==α, y_init_values))
    for y_init in y_init_values
        plot!(residual_plot, t_plot, y.(t_plot; A=y_init), linestyle=:dash, linecolor=:grey, label="")
    end

    # Linear approximation function (continuous and linear on all pieces)
    Δt = t_array[2] - t_array[1]
    function y_approx_func_linear(t)
        index_cont = 1 + (t - t_array[1])/Δt # Add 1 to account for Julia indexing from 1
        i = convert(Int64, floor(index_cont))
        (i == length(t_array)) && return y_approx[end]
        return (t - t_array[i])*(y_approx[i+1] - y_approx[i])/(t_array[i+1] - t_array[i]) +  y_approx[i]
    end

    residual_approx = α .+ residual_integral.(t_plot, f, y_approx_func_linear)
    residual = residual_approx - y_approx_func_linear.(t_plot)
    plot!(
        residual_plot, t_plot, residual_approx,
        linecolor=:orange, label=""
    )

    display(residual_plot)
end

function residual_integral(t::Float64, f, y_approx_func; N=100)
    τ_values = range(0, t, N+1) |> collect
    dτ = τ_values[2] - τ_values[1]
    return sum(f(τ, y_approx_func(τ))*dτ for τ in τ_values)
end