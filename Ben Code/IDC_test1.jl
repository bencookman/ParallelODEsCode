using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra, LaTeXStrings

include("ProjectTools.jl")

using .ProjectTools

"""
Algorithm from https://doi.org/10.1137/09075740X
This code is horrible and very un-Julia for the sake of matching the algorithm
in the above paper as closely as possible. A refactored version will be made in
time. Note indexing used for vectors in given algorithm starts at 0 but Julia
starts at 1.
This algorithm is very specific to equidistant nodes. Not so specific to choice
of integrating scheme.
"""
function IDC(f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N+1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, M)
    η = Array{Float64, 1}(undef, N+1)
    η_old = η
    η_old[1] = α

    S = compute_integration_matrix(M; integral_resolution=20)

    for j in 0:(J-1)
        # Prediction loop
        for m in 0:(M-1)
            k = j*M + m + 1
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        for l in 1:M
            for m in 0:(M-1)
                k = j*M + m + 1
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*sum(S[m+1, i+1]*f(t[j*M + i + 1], η_old[j*M + i + 1]) for i in 0:M)
                # println(η[k] - η_old[k])
            end
            η_old = η
        end
    end

    return η
end

function IDC_test_func(f, y, α, t_end, p, N_array)
    t_plot = range(0, t_end, 1000) |> collect
    η_plot = y.(t_plot)
    Δt_array = t_end./N_array
    err_array = []

    plot_func = plot(
        t_plot, η_plot,
        ylimits=(0.2, 0.8), xlabel=L"t", ylabel=L"y"
    )
    for N in N_array
        t_in = range(0, t_end, N+1) |> collect
        η_out = IDC(f, 0, t_end, α, N, p)
        η_exact = y.(t_in)
        plot!(
            plot_func, t_in, η_out,
            label=latexstring("N = $(N)")
        )

        # Global error was weird, so try end local error
        err = err_abs(η_exact, η_out)[end]
        push!(err_array, err)
    end

    err_order_1_array = Δt_array
    err_order_p_array = Δt_array.^p # Take error constant C = 1
    plot_err = plot(
        Δt_array, err_order_p_array,
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel=L"||E||",
        linestyle=:dash, label=L"1\cdot (\Delta t)^%$p",
        key=:bottomright
    )
    plot!(
        plot_err, Δt_array, err_order_1_array,
        linestyle=:dash, label=L"1\cdot (\Delta t)^1"
    )
    plot!(
        plot_err, Δt_array, err_array,
        markershape=:circle, markerstrokealpha=0, label=latexstring("Approximate solution at \$p = $(p)\$")
    )
    savefig(plot_err, "Ben Code/output/tests/test19.png")
end

"""
Taken from page 53 of Numerical Methods for ODEs by J C Butcher. This is just a
test to see if it works at the specified order of accuracy.
"""
function IDC_test_1()
    α = 0.4
    t_end = 1.0
    p = 5
    N_array = (p-1).*[6*10^i for i in 1:5]

    grad_func(t, y) = (y-2t*y^2)/(1+t)
    exact_func(t) = (1+t)/(t^2+1/α)
    IDC_test_func(grad_func, exact_func, α, t_end, p, N_array)
end

"""
Another test
doi: 10.1137/09075740X
"""
function IDC_test_2()
    α = 1.0
    t_end = 5.0
    p = 5
    N_array = (p-1).*[6*10^i for i in 1:5]

    grad_func(t, y) = 4t*sqrt(y)
    exact_func(t) = (1 + t^2)^2
    IDC_test_func(grad_func, exact_func, α, t_end, p, N_array)
end