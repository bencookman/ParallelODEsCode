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
            η[k + 1] = η[k]
                + Δt*(f(t[k], η[k]) - f(t[k], η_old[k]))
                + Δt*sum(S[m+1, i+1]*f(t[j*M + i + 1], η_old[j*M + i + 1]) for i in 0:M)
        end
        η_old = η
    end

    return η
end

function IDC_test()
    t_end = 1.0
    α = 0.4
    N = 60
    p = 4

    # Taken from page 53 of Numerical Methods for ODEs by J C Butcher
    test_func(t, u) = (u-2t*u^2)/(1+t)
    η_correct(t) = (1+t)/(t^2+1/α)

    η_out = IDC(test_func, 0, t_end, α, N, p)
    t_in = range(0, t_end, N+1) |> collect
    t_plot = range(0, t_end, 1000) |> collect

    my_plot = plot(t_in, η_out)
    plot!(my_plot, t_plot, η_correct.(t_plot), ylimits=(0.2, 0.8))
end
