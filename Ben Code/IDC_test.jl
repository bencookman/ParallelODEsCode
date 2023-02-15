using
    Dates,
    Plots,
    BenchmarkTools,
    Statistics,
    LinearAlgebra,
    LaTeXStrings

include("ProjectTools.jl")

using .ProjectTools

function test_IDC_SDC(
    ODE_test_system::ODETestSystem,
    p, N_uniform, N_legendre, orders_to_plot
)
    @unpack_ODETestSystem ODE_test_system
    @unpack_ODESystem ODE_system
    Δt_array_uniform = (t_e - t_s)./N_uniform
    Δt_array_legendre = (t_e - t_s)./N_legendre
    err_IDC_array = []
    err_SDC_array = []
    # err_RK_array = []

    S_uniform = integration_matrix_uniform(p - 1)
    S_legendre = integration_matrix_legendre(p)
    η_exact = y(t_e)
    for i in axes(N_legendre)[1]
        (_, η_approx_IDC) = IDC_FE_correction_levels(ODE_system, N_uniform[i], p, S_uniform)
        η_out_IDC = real(η_approx_IDC[end, end])
        err_IDC = err_rel(η_exact, η_out_IDC)
        push!(err_IDC_array, (err_IDC <= 0.0) ? 1.0 : err_IDC)

        (_, η_approx_SDC) = SDC_FE_correction_levels(ODE_system, N_legendre[i], p, S_legendre)
        η_out_SDC = real(η_approx_SDC[end, end])
        err_SDC = err_rel(η_exact, η_out_SDC)
        push!(err_SDC_array, (err_SDC <= 0.0) ? 1.0 : err_SDC)

        # (_, η_approx_RK) = RK4_standard(ODE_system, N)
        # η_out_RK = real(η_approx_RK[end])
        # err_RK = err_rel(η_exact, η_out_RK)
        # push!(err_RK_array, (err_RK <= 0.0) ? 1.0 : err_RK)
    end

    plot_err = plot(
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel="||E||",
        key=:bottomright, size=(1600, 1200), thickness_scaling=2.0
    )
    plot!(
        plot_err,
        Δt_array_uniform, err_IDC_array,
        markershape=:circle, label=latexstring("Solution approximated with IDC at \$p = $(p)\$"), color = :blue,
    )
    plot!(
        plot_err,
        Δt_array_legendre, err_SDC_array,
        markershape=:square, label=latexstring("Solution approximated with SDC at \$p = $(p)\$"), color = :red,
    )
    # plot!(
    #     plot_err,
    #     Δt_array, err_RK_array,
    #     markershape=:square, label="Solution approximated with RK4", color = :red
    # )
    for order in orders_to_plot
        err_order_array = Δt_array_legendre.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array_legendre, err_order_array,
            linestyle=:dash, label=L"1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

function test_RIDC()
    N_array = 10:3:52
    p = 8
    S = integration_matrix_uniform(p - 1)

    @unpack_ODETestSystem Butcher_p53_system
    @unpack_ODESystem ODE_system

    orders_to_plot = [p - 1, p]
    Δt_array = (t_e - t_s)./N_array
    err_array = []
    y_exact_end = y(t_e)
    for N in N_array
        (_, y_out) = RIDC_RK2_trapeoid_sequential(ODE_system, N, N, p, S)
        y_out_end = real(y_out[end])
        err = err_rel(y_exact_end, y_out_end)
        push!(err_array, err)
    end

    plot_err = plot(
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel="||E||",
        key=:bottomright, size=(1600, 1200), thickness_scaling=2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape=:circle, label="Solution approximated with RIDC8-RK2", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle=:dash, label=L"1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

function IDC_test_implicit_correction_levels()
    p = 4
    N_array = (1:3:100) .* (p - 1)
    S = integration_matrix_uniform(p - 1)

    orders_to_plot = 1:p

    ∂f∂y(t, y) = 4
    @unpack_ODETestSystem stiff_system_1
    @unpack_ODESystem ODE_system
    Δt_array = (t_e - t_s)./N_array
    levels_err_array = Array{Float64, 2}(undef, p, length(N_array))

    η_exact = y(t_e)
    for (i, N) in enumerate(N_array)
        (_, η_approx) = IDC_Euler_implicit_1D_correction_levels(
            ODE_system,
            ∂f∂y,
            N, p, S
        )
        η_out = real(η_approx[end, :])
        for l in axes(η_out)[1]
            err = err_rel(η_exact, η_out[l])
            levels_err_array[l, i] = err
        end
    end
    plot_err = plot(
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel="||E||",
        key=:bottomright, size=(1600, 1200), thickness_scaling=2.0
    )
    for l in axes(levels_err_array)[1]
        plot!(
            plot_err,
            Δt_array, levels_err_array[l, :],
            markershape=:circle, label=latexstring("Solution approximated with IDC-BE at level \$l = $l\$")
        )
    end
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle=:dash, label=L"1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

function IDC_test_implicit()
    p = 4
    N_array = (1:3:100) .* (p - 1)
    S = integration_matrix_uniform(p - 1)

    orders_to_plot = 1:p

    ∂f∂y(t, y) = 4
    @unpack_ODETestSystem stiff_system_1
    @unpack_ODESystem ODE_system
    Δt_array = (t_e - t_s)./N_array
    err_array = Vector{Float64}(undef, length(N_array))

    η_exact = y(t_e)
    for (i, N) in enumerate(N_array)
        (_, η_approx) = backward_Euler_1D(
            ODE_system,
            ∂f∂y,
            N
        )
        η_out = real(η_approx[end, end])
        err = err_rel(η_exact, η_out)
        err_array[i] = err
    end
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = L"Δt", ylabel = "||E||",
        key = :bottomright, size = (1600, 1200), thickness_scaling = 2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "Solution approximated with BE",
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

function IDC_test_1()
    p = 4
    K = 5
    J = collect(1:3:100)
    IDC_test_func(Butcher_p53_system, p, K, K.*J)
end

function IDC_test_2()
    p = 8
    # K = 5
    J = collect(1:40)
    orders_to_plot = [p - 1, p]
    IDC_test_func(sqrt_system, p, (p - 1) .* J, (p + 1) .* J, orders_to_plot)
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
        S = integration_matrix_uniform(sum_resolution)
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


function predict_group(
    η_start, t_group,
    f, value_type,
    M, Δt
)
    η_group = zeros(value_type, M + 1)
    η_group[1] = η_start
    for m in 1:M
        η_group[m + 1] = η_group[m] + Δt*f(t_group[m], η_group[m])
    end
    return η_group
end

function correct_group(
    η_start, t_group, η_old_group,
    f, value_type,
    M, S, Δt
)
    η_group = zeros(value_type, M + 1)
    η_group[1] = η_start
    for m in 1:M
        ∫fₘ = dot(S[m, :], f.(t_group, η_old_group))
        η_group[m + 1] = η_group[m] + Δt*(f(t_group[m], η_group[m]) - f(t_group[m], η_old_group[m])) + Δt*∫fₘ
    end
    return η_group
end

"""
A slightly modified IDC method where we do the prediction and correction levels
over the whole time domain sequentially.
This allows for parallelisation 'over the groups'.
"""
function IDC_across_groups_sequential(
    ODE_system::ODESystem,
    J, number_corrections, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    p = number_corrections + 1
    M = p - 1
    N = M*J
    t = range(t_s, t_e, N + 1)  # Uniform time nodes
    Δt = (t_e - t_s)/N
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    # Prediction
    for j in 0:(J - 1)
        I = (j*M + 1):((j + 1)*M + 1)
        η[I] .= predict_group(η[I[1]], t[I],f, typeof(y_s), M, Δt)
    end
    # Correction
    for _ in 1:number_corrections
        η_old = copy(η)
        for j in 0:(J - 1)
            I = (j*M + 1):((j + 1)*M + 1)
            η[I] .= correct_group(η[I[1]], t[I], η_old[I], f, typeof(y_s), M, S, Δt)
        end
    end

    return (t, η)
end

function IDC_across_groups_parallel(
    ODE_system::ODESystem,
    J, number_corrections, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    p = number_corrections + 1
    M = p - 1
    N = M*J
    t = range(t_s, t_e, N + 1)  # Uniform time nodes
    Δt = (t_e - t_s)/N
    η = zeros(typeof(y_s), N + 1, p)
    η[1, :] .= y_s

    (J <= (number_corrections + 1)) && throw(error("too few groups to run"))

    # Startup
    for j in 0:number_corrections
        I = (j*M + 1):((j + 1)*M + 1)
        η[I, 1] .= predict_group(η[I[1], 1], t[I], f, typeof(y_s), M, Δt)
    end
    for l in 1:number_corrections
        for j in 0:(number_corrections - l)
            I = (j*M + 1):((j + 1)*M + 1)
            η[I, l + 1] .= correct_group(η[I[1], l + 1], t[I], η[I, l], f, typeof(y_s), M, S, Δt)
        end
    end

    # Corrections can now be run at the same time (here, before) as prediction
    for j in (number_corrections + 1):(J - 1)
        for l in number_corrections:-1:1
            j_level = j - l
            I = (j_level*M + 1):((j_level + 1)*M + 1)   # Group changes with correction level
            η[I, l + 1] .= correct_group(η[I[1], l + 1], t[I], η[I, l], f, typeof(y_s), M, S, Δt)
        end
        I = (j*M + 1):((j + 1)*M + 1)
        η[I, 1] .= predict_group(η[I[1], 1], t[I], f, typeof(y_s), M, Δt)
    end

    # To terminate we run in sequence again
    for l in 1:number_corrections
        for j in (J - l):(J - 1)
            I = (j*M + 1):((j + 1)*M + 1)
            η[I, l + 1] .= correct_group(η[I[1], l + 1], t[I], η[I, l], f, typeof(y_s), M, S, Δt)
        end
    end

    return (t, η[:, end])
end

function test_IDC_across_groups()
    J_array = 4:20
    number_corrections = 2
    p = number_corrections + 1
    orders_to_plot = 1:p

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    S = integration_matrix_uniform(p - 1)
    N_array = J_array*(p - 1)
    Δt_array = (t_e - t_s)./N_array
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (_, y_out) = IDC_across_groups_parallel(ODE_system, J, number_corrections, S)
        y_out_end = real(y_out[end])
        err = err_rel(y_exact_end, y_out_end)
        push!(err_array, err)
    end

    plot_err = plot(
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel="||E||",
        key=:bottomright, size=(1600, 1200), thickness_scaling=2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape=:circle, label="Solution approximated with \'IDC across the groups\'", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle=:dash, label=L"1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end
