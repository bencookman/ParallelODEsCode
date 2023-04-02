using
    Dates,
    Plots,
    BenchmarkTools,
    Statistics,
    LinearAlgebra,
    LaTeXStrings,
    FastGaussQuadrature,
    PyCall,
    ProgressMeter,
    Measures

include("ProjectTools.jl")

using .ProjectTools

const MATPLOTLIB = pyimport("matplotlib")
const RCPARAMS = PyDict(MATPLOTLIB["rcParams"])
RCPARAMS["mathtext.fontset"] = "cm"
RCPARAMS["xtick.major.pad"] = 10

function test_RK()
    p = 4
    orders_to_plot = p

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    N_array = 5:40
    Δt_array = []
    err_array = []

    y_exact_end = y(t_e)
    for N in N_array
        (t, y_out) = RK2_midpoint(ODE_system, N)
        y_out_end = real(y_out[end])
        err = abs(y_exact_end - y_out_end)/abs(y_exact_end)
        # err = err_norm(y_exact_end, y_out_end, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(Δt_array, t[2] - t[1])
        push!(err_array, err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = :bottomright,
        size = (900, 900), thickness_scaling = 1.0,
        guidefontsize = 10, tickfontsize = 8, legendfontsize = 8,
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "FE", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t)^{%$order}"
        )
    end
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/convergence/$dtstring-convergence-FE.png"
    savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    println(err_array)
    for i in axes(err_array, 1)[2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_array[i]) - log(Δt_array[i - 1]))
        println(order)
    end
end

function test_multiple_RK_methods()
    @unpack_ODETestSystem Butcher_p53_system
    @unpack_ODESystem ODE_system

    N_array = 5:40
    one_over_N_array = 1.0./N_array

    methods_array = [RK1_forward_euler, RK2_Heuns, RK3_Kutta, RK4_standard]
    err_array = [[] for _ in axes(methods_array, 1)]
    y_exact_end = y(t_e)
    for N in N_array
        for (i, method) in enumerate(methods_array)
            (t, y_out) = method(ODE_system, N)
            y_out_end = real(y_out[end])
            err = abs(y_exact_end - y_out_end)/abs(y_exact_end)
            # err = err_norm(y_exact_end, y_out_end, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
            push!(err_array[i], err)
        end
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        size = (1750, 1000), thickness_scaling = 4.0,
        guidefontsize = 10, tickfontsize = 8, legendfontsize = 8,
        legend = (1, 0),
        topmargin = 3mm
    )
    orders_to_plot = [1, 2, 3, 4]
    coeffs = [0.7, 0.02, 0.025, 0.005]
    colours = [:red, :orange, :green, :blue]
    for (i, order) in enumerate(orders_to_plot)
        err_order_array = coeffs[i].*one_over_N_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, one_over_N_array, err_order_array,
            linestyle = :dash, label = L"N^{-%$order}\cdot %$(coeffs[i])",
            color = colours[i]
        )
    end
    method_labels = ["FE", "RK2 Heun's", "RK3 Kutta's", "RK4 Standard"]
    for (i, errs) in enumerate(err_array)
        plot!(
            plot_err,
            one_over_N_array, errs,
            markershape = :circle, label = method_labels[i], color = colours[i]
        )
    end
    xtick_values = 1.0./[5, 10, 20, 30, 40]
    xtick_strings = ["1/5", "1/10", "1/20", "1/30", "1/40"]
    xticks!(plot_err, xtick_values, xtick_strings)
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    # println(err_array)
    # for i in axes(err_array, 1)[2:end]
    #     order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_array[i]) - log(Δt_array[i - 1]))
    #     println(order)
    # end
end

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

function test_IDC()
    # J_array = 1:3:100
    J_array = 1:30
    number_corrections = 3
    p = number_corrections + 1
    # M = p - 1
    M = ceil(Int64, number_corrections/2) + 1
    orders_to_plot = [p - 1, p]
    # S = integration_matrix_uniform_RK4(M)
    S = integration_matrix_uniform(M)

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    Δt_array = []
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        # S = integration_matrix_uniform(J)
        (t, y_out) = IDC_FE(ODE_system, number_corrections, S, J)
        y_out_end = real(y_out[end, end])
        err = err_norm(y_exact_end, y_out_end, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(Δt_array, t[2] - t[1])
        push!(err_array, err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = L"Δt", ylabel = "||E||",
        key = :bottomright, size = (1000, 750), thickness_scaling = 2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "IDC approximation", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t)^{%$order}"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array, 1)[2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_array[i]) - log(Δt_array[i - 1]))
        println(order)
    end
end

function test_IDC_levels()
    N_array = 6:6:120
    orders_to_plot = 1:4


    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    Δt_array = []
    err_array = [[] for _ in 1:4]
    y_exact_end = y(t_e)
    for N in N_array
        # FE
        (t, y_out) = RK1_forward_euler(ODE_system, N)    
        y_out_end_level = real(y_out[end])
        err_level = err_norm(y_exact_end, y_out_end_level, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(err_array[1], err_level)

        # 1 CORRECTION
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[2] # 1 correction requires 2 quadrature nodes
        (t, y_out) = IDC_FE(ODE_system, 1, S, N)
        y_out_end_level = real(y_out[end, end])
        err_level = err_norm(y_exact_end, y_out_end_level, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(err_array[2], err_level)

        # 2 CORRECTIONS
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[3] # 2 corrections requires 3 quadrature nodes
        (t, y_out) = IDC_FE(ODE_system, 2, S, Int64(N/2))
        y_out_end_level = real(y_out[end, end])
        err_level = err_norm(y_exact_end, y_out_end_level, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(err_array[3], err_level)

        # 3 CORRECTIONS
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[4] # 3 corrections requires 4 quadrature nodes
        (t, y_out) = IDC_FE(ODE_system, 3, S, Int64(N/3))
        y_out_end_level = real(y_out[end, end])
        err_level = err_norm(y_exact_end, y_out_end_level, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(err_array[4], err_level)

        push!(Δt_array, (t_e - t_s)/N)
    end

    plot_err = plot( 
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = :bottomright, size = (1750, 1000), thickness_scaling = 4.0,
        legend = (1, 0),
        margin = 6mm,
    )
    colours = [:red, :orange, :green, :blue]
    log_const = [1, 0.5, 0.05, 0.01]
    for (i, order) in enumerate(orders_to_plot)
        err_order_array = log_const[i].*(Δt_array).^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"(\Delta t)^{%$order} \cdot %$(log_const[i])",
            color = colours[order]
        )
    end
    # cycle_index(index; max_index = 4) = (index - 1)%max_index + 1
    level_label = ["FE", "1 Correction", "2 Corrections", "3 Corrections"]
    for level in 1:4
        plot!(
            plot_err,
            Δt_array, err_array[level],
            markershape = :circle, label = level_label[level], color = colours[level]
        )
    end
    # xtick_values = 1.0./[5, 10, 20, 30, 40]
    # xtick_strings = ["1/5", "1/10", "1/20", "1/30", "1/40"]
    # xticks!(plot_err, xtick_values, xtick_strings)
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    # for i in axes(err_array, 1)[2:end]
    #     order = (log(err_array[i]) - log(err_array[i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
    #     println(order)
    # end
end


function test_SDC_single()
    N_array = 5:40      # Number of time steps
    one_over_N_array = 1.0./N_array
    number_corrections = 3
    p = number_corrections + 1

    orders_to_plot = [p]

    @unpack_ODETestSystem Butcher_p53_system
    @unpack_ODESystem ODE_system

    err_array = []
    y_exact_end = y(t_e)
    for N in N_array
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[N + 1]
        # S = INTEGRATION_MATRIX_ARRAY_LEGENDRE[N - 1]
        # S = INTEGRATION_MATRIX_ARRAY_LOBATTO[N + 1]
        (t, y_out) = IDC_FE_single(ODE_system, number_corrections, S, N)
        # (t, y_out) = SDC_FE_legendre_single(ODE_system, number_corrections, S, N)
        # (t, y_out) = SDC_FE_lobatto_single(ODE_system, number_corrections, S, N)
        y_out_end = real(y_out[end, end])
        err = err_norm(y_exact_end, y_out_end, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(err_array, err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        key = :bottomright, size = (800, 600), thickness_scaling = 1.0,
        legend = (1, 0),
        topmargin = 3mm,
    )
    for order in orders_to_plot
        err_order_array = (one_over_N_array).^order # Taking error constant = 1 always
        plot!(
            plot_err, one_over_N_array, err_order_array,
            linestyle = :dash, label = L"1 \cdot N^{-%$order}"
        )
    end
    plot!(
        plot_err,
        one_over_N_array, err_array,
        markershape = :circle, label = "", color = :blue
    )
    xtick_values = 1.0./[5, 10, 20, 30, 40]
    xtick_strings = ["1/5", "1/10", "1/20", "1/30", "1/40"]
    xticks!(plot_err, xtick_values, xtick_strings)
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array, 1)[2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
        println(order)
    end
end

function test_SDC_single_levels()
    N_array = 5:40      # Number of time steps
    one_over_N_array = 1.0./N_array
    number_corrections = 3
    p = number_corrections + 1

    orders_to_plot = 1:p

    @unpack_ODETestSystem Butcher_p53_system
    @unpack_ODESystem ODE_system

    err_array = [[] for _ in 1:(number_corrections + 1)]
    y_exact_end = y(t_e)
    for N in N_array
        # S = INTEGRATION_MATRIX_ARRAY_LEGENDRE[N - 1]
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[N + 1]
        # (t, y_out) = SDC_FE_legendre_single(ODE_system, number_corrections, S, N)
        (t, y_out) = SDC_FE_lobatto_single(ODE_system, number_corrections, S, N)
        for level in 1:(number_corrections + 1)
            y_out_end_level = real(y_out[end, level])
            err_level = err_norm(y_exact_end, y_out_end_level, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
            push!(err_array[level], err_level)
        end
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        key = :bottomright, size = (1750, 1000), thickness_scaling = 4.0,
        legend = (1, 0),
        margin = 6mm,
    )
    colours = [:red, :orange, :green, :blue]
    log_const = [1, 0.5, 0.2, 0.05]
    for (i, order) in enumerate(orders_to_plot)
        err_order_array = log_const[i].*(one_over_N_array).^order # Taking error constant = 1 always
        plot!(
            plot_err, one_over_N_array, err_order_array,
            linestyle = :dash, label = L"N^{-%$order} \cdot %$(log_const[i])",
            color = colours[order]
        )
    end
    # cycle_index(index; max_index = 4) = (index - 1)%max_index + 1
    level_label = ["Prediction", "Correction #1", "Correction #2", "Correction #3"]
    for level in 1:(number_corrections + 1)
        plot!(
            plot_err,
            one_over_N_array, err_array[level],
            markershape = :circle, label = level_label[level], color = colours[level]
        )
    end
    xtick_values = 1.0./[5, 10, 20, 30, 40]
    xtick_strings = ["1/5", "1/10", "1/20", "1/30", "1/40"]
    xticks!(plot_err, xtick_values, xtick_strings)
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    # for i in axes(err_array, 1)[2:end]
    #     order = (log(err_array[i]) - log(err_array[i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
    #     println(order)
    # end
end

function test_SDC()
    J_array = 1:20
    number_corrections = 3
    # p = number_corrections + 1
    p = 4*(number_corrections + 1)
    # M = p - 1
    # M = 2*number_corrections + 4
    M = 4*(number_corrections + 1) + 1
    # M = ceil(Int64, number_corrections/2) + 1
    orders_to_plot = [p - 1, p]
    # S = integration_matrix_legendre(p)
    S = integration_matrix_legendre_RK4(M - 1)
    # S = integration_matrix_lobatto(M + 1)
    # S = integration_matrix_lobatto_RK4(p)

    @unpack_ODETestSystem stiff_system_1
    @unpack_ODESystem ODE_system

    Δt_array = []
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (t, y_out) = SDC_RK4_legendre(ODE_system, number_corrections, S, J)
        y_out_end = real(y_out[end, end])
        err = err_norm(y_exact_end, y_out_end, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(Δt_array, t[2] - t[1])
        push!(err_array, err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = L"Δt", ylabel = "||E||",
        key = :bottomright, size = (1000, 750), thickness_scaling = 2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "SDC", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t)^{%$order}"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array, 1)[2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_array[i]) - log(Δt_array[i - 1]))
        println(order)
    end
end

function test_RIDC()
    number_corrections = 1
    p = 4*(number_corrections + 1)
    J_array = 1:20
    # K_array = (p - 1):3:100
    # J = 1
    K = 20
    S = integration_matrix_uniform_RK4(p - 1)
    # M_levels = [2*level - 1 for level in 2:(number_corrections + 1)]
    # S_levels = [integration_matrix_uniform(M_levels[level - 1]) for level in 2:(number_corrections + 1)]
    orders_to_plot = [p - 1, p]

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    Δt_array = []
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (t, y_out) = RIDC_RK4(ODE_system, number_corrections, S, J, K)
        y_out_end = real(y_out[end, end])
        err = err_norm(y_exact_end, y_out_end, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(Δt_array, t[2] - t[1])
        push!(err_array, err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = L"Δt", ylabel = "||E||",
        key = :bottomright, size = (1000, 750), thickness_scaling = 2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "Solution approximated with RIDC", color = :blue,
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

    ## PRINT ORDER
    for i in axes(err_array, 1)[2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_array[i]) - log(Δt_array[i - 1]))
        println(order)
    end
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

function test_IDC_across_groups_reduced_stencil()
    J_array = 1:20
    number_corrections = 15
    number_final_level_nodes = 1

    p = number_corrections + 1
    orders_to_plot = 1:p

    @unpack_ODETestSystem Butcher_p53_system
    @unpack_ODESystem ODE_system

    S_levels = []
    lobatto_nodes_levels = [gausslobatto(level)[1] for level in 2:p]
    push!(lobatto_nodes_levels, gausslobatto(1 + number_final_level_nodes)[1])
    for level in 2:p
        S = integration_matrix(lobatto_nodes_levels[level - 1], lobatto_nodes_levels[level])
        push!(S_levels, S)
    end

    Δt_group_array = (t_e - t_s)./J_array
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (_, y_out) = RSDC_FE_lobatto_reduced_stencil(
            ODE_system,
            J, number_corrections, S_levels, number_final_level_nodes
        )
        y_out_end = real(y_out[end][end])
        err = err_rel(y_exact_end, y_out_end)
        push!(err_array, err)
    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = L"Δt", ylabel = "||E||",
        key = :bottomright, size = (1000, 750), thickness_scaling = 2.0, legendfontsize = 6, legendposition = :none
    )
    plot!(
        plot_err,
        Δt_group_array, err_array,
        markershape = :circle, label = "Solution approximated with \'RSDC across the groups\'", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_group_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_group_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t_{\text{group}})^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array)[1][2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_group_array[i]) - log(Δt_group_array[i - 1]))
        println(order)
    end
end

function test_RSDC()
    J_array = 1:20
    number_corrections = 1
    p = 4*(number_corrections + 1)
    orders_to_plot = [p - 1, p]

    @unpack_ODETestSystem Butcher_p53_system
    @unpack_ODESystem ODE_system

    # S = integration_matrix_uniform_RK4(p - 1)
    S = integration_matrix_lobatto_RK4(p)
    Δt_group_array = (t_e - t_s)./J_array
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (_, y_out) = RSDC_RK4_lobatto(
            ODE_system,
            number_corrections, S, J
        )
        y_out_end = real(y_out[end, end])
        err = err_norm(y_exact_end, y_out_end, 2)/norm(y_exact_end, 2)  # Relative 2-norm error
        push!(err_array, err)
    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = L"Δt", ylabel = "||E||",
        key = :bottomright, size = (1000, 750), thickness_scaling = 2.0, legendfontsize = 6, legendposition = :none
    )
    plot!(
        plot_err,
        Δt_group_array, err_array,
        markershape = :circle, label = "Solution approximated with \'RSDC across the groups\'", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_group_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_group_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t_{\text{group}})^%$order"
        )
    end

    ## PRINT ORDER
    for i in axes(err_array)[1][2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_group_array[i]) - log(Δt_group_array[i - 1]))
        println(order)
    end

    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
    plot_err
end

