using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra, LaTeXStrings

include("ProjectTools.jl")

using .ProjectTools

# function IDC_test_func(f, y, α, t_end, p, N_array, K)
function IDC_test_func(ODE_system::ODESystem, p, K, N_array)
    @unpack_ODESystem ODE_system
    Δt_array = (t_e - t_s)./N_array
    err_array = []

    S = integration_matrix_equispaced(p - 1)
    for N in N_array
        (t_in, η_approx) = RIDC_FE_sequential(S, f, t_s, t_e, y_s, N, K, p)
        η_out = η_approx[:, end]

        η_exact = y.(t_in)
        err = err_rel(η_exact, η_out)[end]
        push!(err_array, (err <= 0.0) ? 1.0 : err)
    end

    plot_err = plot(
        Δt_array, err_array,
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel="||E||",
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
    fname = "Ben Code/output/convergence/$dtstring-convergence-RIDC_FE_sequential.png"
    savefig(plot_err, fname)
    # display(plot_err)
end


""" Taken from page 53 of Numerical Methods for ODEs by J C Butcher """
const Butcher_p53_system = ODESystem(
    (t, y) -> (y - 2t*y^2)/(1 + t),
    t -> (1 + t)/(t^2 + 1/0.4),
    0.4,
    1.0
)
""" https://doi.org/10.1137/09075740X """
const sqrt_system = ODESystem(
    (t, y) -> 4t*sqrt(Complex(y)),
    t -> (1 + t^2)^2,
    1.0,
    5.0
)
const cube_system = ODESystem(
    (t, y) -> t^3,
    t -> 0.25*t^4 + 2.0,
    2.0,
    5.0
)

function IDC_test_1()
    p = 4
    K = 3
    J = collect(1:3:100)
    IDC_test_func(Butcher_p53_system, p, K, K.*J)
end

function IDC_test_2()
    p = 6
    K = 100
    J = collect(1:19)
    IDC_test_func(sqrt_system, p, K, K.*J)
end

function IDC_test_3()
    p = 5
    # N_array = (p - 1).*collect(2:3:100)
    N_array_single = collect(4:15)

    IDC_test_func(cube_system, p, N_array_single)
end


function integration_matrix_test()
    # Set up test
    # t_end = 1.0
    # f(t) = cos(t)
    # integral_exact = sin(t_end)
    # t_end = 1
    # f(t) = cos(t)^2
    # integral_exact = t_end/2 + sin(2*t_end)/4
    t_end = 1
    f(t) = sqrt(t)
    integral_exact = 2*(t_end)^(1.5)/3
    # t_end = 0.001
    # f(t) = cos(t)*exp(sin(t))
    # integral_exact = exp(sin(t_end)) - 1

    # Do test
    integral_approximations = Array{Float64, 1}(undef, 0)
    sum_resolutions = 2:2:30
    for sum_resolution in sum_resolutions
        t_sample = 0:(sum_resolution + 1) |> collect
        f_sample = f.(t_sample)
        S = integration_matrix_equispaced(sum_resolution)
        integral_approx = sum(S[1, i] .* f_sample[i] for i in 1:(sum_resolution+1))

        println(integral_approx)
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
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/tests/int-matrix-new-err-cos-2-10-$dtstring.png"
    savefig(test_plot, fname)
end