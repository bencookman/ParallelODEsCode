using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra, LaTeXStrings

include("ProjectTools.jl")

using .ProjectTools

err_abs(exact, approx) = abs.(exact - approx)
err_cum(exact, approx) = [sum(err_abs(exact[1:i], approx[1:i])) for i in 1:length(approx)]


"""
Algorithm from https://doi.org/10.1137/09075740X
This code is horrible and very un-Julia for the sake of matching the algorithm
in the above paper as closely as possible. A refactored version will be made in
time. Note indexing used for vectors in given algorithm starts at 0 but Julia
starts at 1.
"""
function IDC(f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N+1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, M)
    η = Array{Float64, 1}(undef, N+1)
    η[1] = α
    η_old = η

    S = compute_integration_matrix(M; integral_resolution=100)

    for j in 0:(J-1)
        # Prediction loop
        for m in 0:(M-1)
            k = j*M + m + 1
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        for _ in 1:M, m in 0:(M-1)
            k = j*M + m + 1
            η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*sum(S[m+1, i+1]*f(t[j*M + i + 1], η_old[j*M + i + 1]) for i in 0:M)
        end
        η_old = η
    end

    return η
end

function IDC_test()
    t_end = 1.0
    α = 0.4
    p = 5
    N_array = (p-1).*[6*10^i for i in 1:5]

    # Taken from page 53 of Numerical Methods for ODEs by J C Butcher
    # Just a test to see if it works at the specified order of accuracy
    test_func(t, u) = (u-2t*u^2)/(1+t)
    η_correct(t) = (1+t)/(t^2+1/α)

    t_plot = range(0, t_end, 1000) |> collect
    η_plot = η_correct.(t_plot)
    Δt_array = t_end./N_array
    err_array = []

    plot_func = plot(
        t_plot, η_plot,
        ylimits=(0.2, 0.8), xlabel=L"t", ylabel=L"y"
    )
    for N in N_array
        t_in = range(0, t_end, N+1) |> collect
        η_out = IDC(test_func, 0, t_end, α, N, p)
        η_exact = η_correct.(t_in)
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
    # savefig(plot_err, "Ben Code/output/issue_example1.png")

    # plot_all = plot(
    #     plot_func, plot_err,
    #     size=(1600, 800), thickness_scaling=2.0,
    #     markerstyle="."
    # )
end
