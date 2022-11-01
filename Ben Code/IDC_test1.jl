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
    η[1] = α
    η_old = η

    println(Δt)

    S = compute_integration_matrix(M; integral_resolution=100)

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
            end
            η_old = η
        end
    end

    return η
end


""" Integral deferred correction with a single group """
function IDC_single(f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N+1) |> collect
    Δt = (b - a)/N
    η = Array{Float64, 1}(undef, N+1)
    η[1] = α
    η_old = η

    println(Δt)
    S = compute_integration_matrix(N; integral_resolution=100)

    # Prediction loop
    for m in 1:N
        η[m + 1] = η[m] + Δt*f(t[m], η[m])
    end
    # Correction loop
    for l in 1:(p-1)
        for m in 1:N
            η[m + 1] = η[m] + Δt*(f(t[m], η[m]) - f(t[m], η_old[m])) + Δt*sum(S[m, i]*f(t[i], η_old[i]) for i in 1:(N+1))
        end
        η_old = η
    end

    return η
end

""" Integral deferred correction with a single group - use matrix """
function IDC_single_matrix(f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N+1) |> collect
    Δt = (b - a)/N
    η = Array{Float64, 2}(undef, p, N+1)
    η[:, 1] .= α

    println(Δt)
    S = compute_integration_matrix(N; integral_resolution=100)

    # Prediction loop
    for m in 1:N
        η[1, m + 1] = η[1, m] + Δt*f(t[m], η[1, m])
    end
    # Correction loop
    for l in 2:p
        η[l, 1] = η[l-1, 1]
        for m in 1:N
            η[l, m + 1] = η[l, m] + Δt*(f(t[m], η[l, m]) - f(t[m], η[l-1, m])) + Δt*sum(S[m, i]*f(t[i], η[l-1, i]) for i in 1:(N+1))
        end
    end

    return η[end, :]
end

""" Integral deferred correction with a single group - use error and residuals """
function IDC_single_error(f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N+1) |> collect
    Δt = (b - a)/N
    η = Array{Float64, 1}(undef, N+1)
    η[1] = α

    S = compute_integration_matrix(N; integral_resolution=1000)

    # Prediction loop
    for m in 1:N
        η[m + 1] = η[m] + Δt*f(t[m], η[m])
    end
    # Correction loop
    for l in 2:3
        δ = Array{Float64, 1}(undef, N + 1)
        δ[1] = 0
        for m in 1:N
            Δϵ = Δt*sum(S[m, i]*f(t[i], η[i]) for i in 1:(N + 1)) - η[m + 1] + η[m] # difference in residuals at m+1 and m
            δ[m + 1] = δ[m] + Δt*(f(t[m], η[m] + δ[m]) - f(t[m], η[m])) + Δϵ
            # l == 3 && println(Δt*(f(t[m], η[m] + δ[m]) - f(t[m], η[m])) + Δϵ)
        end
        η .+= δ
    end

    return η
end

""" Integral deferred correction with a single group - use polynomial"""
function IDC_single_poly(f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N+1) |> collect
    Δt = (b - a)/N
    η = Array{Float64, 1}(undef, N+1)
    η[1] = α

    # Prediction loop
    for m in 1:N
        η[m + 1] = η[m] + Δt*f(t[m], η[m])
    end
    # Correction loop
    for l in 2:p
        δ = Array{Float64, 1}(undef, N + 1)
        δ[1] = 0

        f_poly(t) = sum(
            f(t[i], η[i])*prod(
                (t - t[k])/(t[i] - t[k])
                for k in 1:(i - 1) ∪ (i + 1):(N + 1)
            )
            for i in 1:(N + 1)
        )
        t_int = range(a, b, 1000) |> collect
        dt = t_int[2] - t_int[1]
        integrals = [sum(f_poly(t)*dt for t in t_int if t <= t[m]) for m in 1:(N + 1)]
        ϵ = integrals .+ α .- η

        for m in 1:N
            Δϵ = ϵ[m + 1] - ϵ[m]
            δ[m + 1] = δ[m] + Δt*(f(t[m], η[m] + δ[m]) - f(t[m], η[m])) + Δϵ
        end
        η .+= δ
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
        η_out = IDC_single_poly(f, 0, t_end, α, N, p)
        η_exact = y.(t_in)
        plot!(
            plot_func, t_in, η_out,
            label=latexstring("N = $(N)")
        )

        # Global error was weird, so try end local error
        err = err_abs(η_exact, η_out)[end]
        push!(err_array, err)
    end

    plot_err = plot(
        Δt_array, err_array,
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel=L"||E||",
        markershape=:circle, label=latexstring("Approximate solution at \$p = $(p)\$"),
        key=:bottomright, size=(1600, 1200), thickness_scaling=2.0
    )
    for order in 1:p
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle=:dash, label=L"1\cdot (\Delta t)^%$order"
        )
    end
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/tests/test-single-poly-$dtstring.png"
    savefig(plot_err, fname)
end

"""
Taken from page 53 of Numerical Methods for ODEs by J C Butcher. This is just a
test to see if it works at the specified order of accuracy.
"""
function IDC_test_1()
    α = 0.4
    t_end = 1.0
    p = 5
    N_array = 2:30

    grad_func(t, y) = (y-2t*y^2)/(1+t)
    exact_func(t) = (1+t)/(t^2+1/α)
    IDC_test_func(grad_func, exact_func, α, t_end, p, N_array)
end

"""
Another test
https://doi.org/10.1137/09075740X
"""
function IDC_test_2()
    α = 1.0
    t_end = 5.0
    p = 5
    N_array = 10:30

    grad_func(t, y) = 4t*sqrt(y)
    exact_func(t) = (1 + t^2)^2
    IDC_test_func(grad_func, exact_func, α, t_end, p, N_array)
end


function integration_matrix_test()
    # Set up test
    t_end = 1.0
    integral_resolution = 1000
    integral_exact = sin(t_end)

    integral_approximations = Array{Float64, 1}(undef, 0)

    # Do test
    sum_resolutions = 1:2:100
    for sum_resolution in sum_resolutions
        t_sample = range(0, t_end, sum_resolution+1) |> collect
        f_sample = cos.(t_sample)
        S = compute_integration_matrix(sum_resolution; integral_resolution=integral_resolution)
        integral_approx = sum(sum(S[:, i] .* f_sample[i] for i in 1:(sum_resolution+1)))/sum_resolution

        push!(integral_approximations, integral_approx)
    end

    # Plot test results
    integral_error = abs.(integral_exact .- integral_approximations)
    Δt_values = t_end./sum_resolutions
    test_plot = plot(
        Δt_values, integral_error,
        xscale=:log10, yscale=:log10, xlabel=L"\Delta t", ylabel=L"||E||",
        size=(1200, 900), thickness_scaling=1.5
    )
end