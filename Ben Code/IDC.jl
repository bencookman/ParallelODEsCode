using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra, LaTeXStrings

include("ProjectTools.jl")

using .ProjectTools

""" Integral deferred correction with a single group - use lagrange polynomial and Newton-Cotes weights """
function IDC(f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N+1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, M)

    η = Array{Float64, 1}(undef, N+1)
    η[1] = α

    for j in 0:(J - 1)
        # Prediction loop
        for m in 0:(M - 1)
            k = j*M + m + 1
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        for l in 2:p
            δ = Array{Float64, 1}(undef, M + 1)
            δ[1] = 0

            f_poly(x) = sum(f(t[j*M + i + 1], η[j*M + i + 1])*prod((x - t[j*M + k + 1])/(t[j*M + i + 1] - t[j*M + k + 1]) for k in 0:M if k != i) for i in 0:M)
            integrals = newton_cotes_integration(t, M, f_poly)
            ϵ = integrals .+ α .- η

            for m in 1:M
                k = j*M + m
                δ[m + 1] = δ[m] + Δt*(f(t[k], η[k] + δ[m]) - f(t[k], η[k])) + ϵ[m + 1] - ϵ[m]
            end
            η[j*M + 1:(j + 1)*M + 1] .+= δ
        end
    end

    return η
end

# Function to calculate closed Newton-Cotes Quadrature weights - FROM BRADLEYS CODE
function newton_cotes_weights(t, n)
    weights = zeros(n+1)
    for j in 1:n+1
        u = union(1:j-1, j+1:n+1)
        coeff = [1, -t[u[1]]] / (t[j] - t[u[1]])
        for l in 2:n
            coeff = ([coeff; 0] - t[u[l]]*[0; coeff]) / (t[j] - t[u[l]])
        end
        evalb = sum((coeff ./ collect(n+1:-1:1)) .* t[end] .^ collect(n+1:-1:1))
        evala = sum((coeff ./ collect(n+1:-1:1)) .* t[1] .^ collect(n+1:-1:1))
        weights[j] = evalb - evala
    end
    return(weights)
end

# Function to approximately integrate a polynomial interpolation of f(u, t) using Newton-Cotes
function newton_cotes_integration(t, n, f_pol)
    int_hat = zeros(length(t))
    for j in 2:length(t)
        sub_t = [i*(t[j]-t[1])/n + t[1] for i in 0:n]
        int_hat[j] = sum(f_pol.(sub_t) .* newton_cotes_weights(sub_t, n))
    end
    return(int_hat)
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
    fname = "Ben Code/output/tests/test-NC-$dtstring.png"
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
    N_array = (p - 1).*[3^i for i in 1:5]

    grad_func(t, y) = (y-2t*y^2)/(1+t)
    exact_func(t) = (1+t)/(t^2+1/α)
    IDC_test_func(grad_func, exact_func, α, t_end, p, N_array)
end
