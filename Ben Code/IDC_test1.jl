using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra

include("ProjectTools.jl")

using .ProjectTools

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

    S = compute_integration_matrix(M; integral_resolution=10)

    for j in 0:(J-1)
        # Prediction loop
        for m in 0:(M-1)
            k = j*M + m + 1
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        for _ in 1:M, m in 0:(M-1)
            k = j*M + m + 1
            η[k + 1] =
                η[k] +
                Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) +
                Δt*sum(S[m+1, i+1]*f(t[j*M + i + 1], η_old[j*M + i + 1]) for i in 0:M)
        end
        η_old = η
    end

    return η
end

function IDC_test()
    t_end = 1.0
    α = 0.4
    N = 60
    p = 2

    # Taken from page 53 of Numerical Methods for ODEs by J C Butcher
    # Just a test to see if it works at the specified order of accuracy
    test_func(t, u) = (u-2t*u^2)/(1+t)
    η_correct(t) = (1+t)/(t^2+1/α)

    t_in = range(0, t_end, N+1) |> collect
    η_out = IDC(test_func, 0, t_end, α, N, p)
    η_exact = η_correct.(t_in)
    t_plot = range(0, t_end, 1000) |> collect
    η_plot = η_correct.(t_plot)


    plot_func = plot(t_in, η_out)
    plot!(my_plot, t_plot, η_plot, ylimits=(0.2, 0.8))

    err_data = error_calculate(η_correct, η_out, 2)
    # savefig(my_plot, "issue_example.png")

    # η_out = IDC((t, u) -> cos(t), 0, t_end, α, N, p)
    # t_in = range(0, t_end, N+1) |> collect
    # my_plot = plot(t_in, η_out, ylimits=(-1, 1))
    # t_plot = range(0, t_end, 1000) |> collect
    # plot!(my_plot, t_plot, sin.(t_plot .+ asin.(0.4)))
    # display(my_plot)
end
