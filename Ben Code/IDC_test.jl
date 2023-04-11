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
    orders_to_plot = [4, 8]

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    N_array = 6:120
    Δt_array = []
    err_array = []

    y_exact_end = y(t_e)
    for N in N_array
        (t, y_out) = RK3_Kutta(ODE_system, N)
        y_out_end = real(y_out[end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array, t[2] - t[1])
        push!(err_array, (err  == 0.0) ? 1.0 : err)
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
        markershape = :circle, label = "RK8", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t)^{%$order}"
        )
    end
    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array, 1)[2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_array[i]) - log(Δt_array[i - 1]))
        println(order)
    end
end

function test_multiple_RK_methods()
    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    N_array = 6:6:111
    # one_over_N_array = 1.0./N_array

    methods_array = [RK4_standard, RK8_Cooper_Verner]
    Δt_array = [[] for _ in axes(methods_array, 1)]
    err_array = [[] for _ in axes(methods_array, 1)]
    y_exact_end = y(t_e)
    for N in N_array
        for (i, method) in enumerate(methods_array)
            (t, y_out) = method(ODE_system, N)
            y_out_end = real(y_out[end])
            err = err_rel_norm(y_exact_end, y_out_end, 2)
            push!(Δt_array[i], t[2] - t[1])
            push!(err_array[i], (err == 0.0) ? 1.0 : err)
        end
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        thickness_scaling = 4.0,
        legend = (1, 0),
    )
    orders_to_plot = [4, 8]
    coeffs = [3e-2, 1e-7]
    order_trunc = [size(N_array, 1), 7]
    colours = [:orange, :blue]
    for (i, order) in enumerate(orders_to_plot)
        Δt_array_trunc = Δt_array[i][1:order_trunc[i]]
        err_order_array = coeffs[i].*Δt_array_trunc.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array_trunc, err_order_array,
            linestyle = :dash, label = L"\sim(\Delta t)^{%$order}",
            color = colours[i]
        )
    end
    method_labels = ["RK4", "RK8"]
    method_markers = [:square, :octagon]
    for (i, err) in enumerate(err_array)
        plot!(
            plot_err,
            Δt_array[i], err,
            markershape = method_markers[i], label = method_labels[i], color = colours[i]
        )
    end
    xtick_values = [10^-0.5, 10^-1.0, 10^-1.5]
    # xtick_strings = ["1/5", "1/10", "1/20", "1/30", "1/40"]
    xticks!(plot_err, xtick_values)
    display(plot_err)
end


function test_IDC()
    J_array = 1:30
    number_corrections = 5
    p = number_corrections + 1
    # p = 2*(number_corrections + 1)
    M = p - 1
    orders_to_plot = [p, p-1, p-2]
    # S = INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS[p]
    S = INTEGRATION_MATRIX_ARRAY_UNIFORM[M + 1]

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    Δt_array = []
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        # (t, y_out) = IDC_RK2_midpoint(ODE_system, number_corrections, S, J)
        # (t, y_out) = IDC_RK2_Heuns(ODE_system, number_corrections, S, J)
        (t, y_out) = IDC_FE(ODE_system, number_corrections, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)  # Relative 2-norm error
        push!(Δt_array, t[2] - t[1])
        push!(err_array, err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = (1, 0), size = (1000, 750), thickness_scaling = 2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "IDC_FE", color = :orange,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t)^{%$order}",
            color = :black
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

function test_IDC_flat_results()
    J_array = 1:10
    number_corrections = 5
    # p = number_corrections + 1
    p = 2*(number_corrections + 1)
    M = p - 1
    orders_to_plot = [p]
    # S = INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS[p]
    S = INTEGRATION_MATRIX_ARRAY_UNIFORM[p]

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    Δt_array = []
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        # (t, y_out) = IDC_RK2_midpoint(ODE_system, number_corrections, S, J)
        (t, y_out) = IDC_RK2_Heuns(ODE_system, number_corrections, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)  # Relative 2-norm error
        push!(Δt_array, t[2] - t[1])
        push!(err_array, err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = (1, 0), size = (1000, 750), thickness_scaling = 4.0
    )
    err_order_array = 1e-4.*Δt_array.^12
    plot!(
        plot_err, Δt_array[1:4], err_order_array[1:4],
        linestyle = :dash, label = L"(\Delta t)^{12} \cdot 10^{-4}",
        color = :black
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :hexagon, label = "IDC12-RK2-Heun's", color = :orange,
    )
    display(plot_err)
end

function test_IDC_RK2()
    J_array = 1:30

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    Δt_array = [[] for _ in 1:4]
    err_array = [[] for _ in 1:4]
    y_exact_end = y(t_e)
    for J in J_array
        # IDC4-RK2-Heuns
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[4]
        (t, y_out) = IDC_RK2_Heuns(ODE_system, 1, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[1], t[2] - t[1])
        push!(err_array[1], err)

        # IDC4-RK2-midpoint
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS[4]
        (t, y_out) = IDC_RK2_midpoint(ODE_system, 1, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[2], t[2] - t[1])
        push!(err_array[2], err)

        # IDC8-RK2-Heuns
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[8]
        (t, y_out) = IDC_RK2_Heuns(ODE_system, 3, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[3], t[2] - t[1])
        push!(err_array[3], err)

        # IDC8-RK2-midpoint
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS[8]
        (t, y_out) = IDC_RK2_midpoint(ODE_system, 3, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[4], t[2] - t[1])
        push!(err_array[4], err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    orders_to_plot = [4, 8]
    log_const = [1, 0.01]
    trunc_plot = [length(J_array), 12]
    order_colour = [:red, :orange]
    for (i, order) in enumerate(orders_to_plot)
        order_Δt_array = Δt_array[2i]
        err_order_array = log_const[i].*(order_Δt_array.^order)
        plot!(
            plot_err, order_Δt_array[1:trunc_plot[i]], err_order_array[1:trunc_plot[i]],
            linestyle = :dash, label = L"\sim(\Delta t)^{%$order}", color = order_colour[i]
        )
    end
    data_color = [:red, :green, :orange, :lime]
    data_shape = [:circle, :square, :circle, :square]
    data_label = ["IDC4-RK2-Heun's", "IDC4-RK2-midpoint", "IDC8-RK2-Heun's", "IDC8-RK2-midpoint"]
    for i in 1:4
        plot!(
            plot_err,
            Δt_array[i], err_array[i],
            markershape = data_shape[i], label = data_label[i], color = data_color[i],
            markersize = 3.5, markerstrokewidth = 0.8
        )
    end

    display(plot_err)
end

function test_IDC12()
    J_array = 1:10

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    Δt_array = [[] for _ in 1:4]
    err_array = [[] for _ in 1:4]
    y_exact_end = y(t_e)
    for J in J_array
        S_1 = INTEGRATION_MATRIX_ARRAY_UNIFORM[12]
        S_2 = INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS[12]

        # IDC12-FE
        (t, y_out) = IDC_FE(ODE_system, 11, S_1, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[1], t[2] - t[1])
        push!(err_array[1], err)

        # IDC12-RK2-Heun's
        (t, y_out) = IDC_RK2_Heuns(ODE_system, 5, S_1, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[2], t[2] - t[1])
        push!(err_array[2], err)

        # IDC12-RK3
        (t, y_out) = IDC_RK3(ODE_system, 3, S_2, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[3], t[2] - t[1])
        push!(err_array[3], err)

        # IDC12-RK4
        (t, y_out) = IDC_RK4(ODE_system, 2, S_2, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[4], t[2] - t[1])
        push!(err_array[4], err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    orders_to_plot = [12]
    log_const = [1]
    trunc_plot = [length(J_array)]
    for (i, order) in enumerate(orders_to_plot)
        order_Δt_array = Δt_array[i]
        err_order_array = log_const[i].*(order_Δt_array.^order)
        plot!(
            plot_err, order_Δt_array[1:trunc_plot[i]], err_order_array[1:trunc_plot[i]],
            linestyle = :dash, label = L"(\Delta t)^{%$order}\cdot %$(log_const[i])", color = :black
        )
    end
    data_color = [:red, :orange, :green, :blue]
    data_shape = [:circle, :hexagon, :ltriangle, :square]
    data_label = ["IDC12-FE", "IDC12-RK2-Heun's", "IDC12-RK3", "IDC12-RK4"]
    for i in 1:4
        plot!(
            plot_err,
            Δt_array[i], err_array[i],
            markershape = data_shape[i], label = data_label[i], color = data_color[i],
            markersize = 3.5, markerstrokewidth = 0.8
        )
    end

    display(plot_err)
end

function test_IDC8()
    J_array = 1:10

    @unpack_ODETestSystem exp_system_1
    @unpack_ODESystem ODE_system

    Δt_array = [[] for _ in 1:3]
    err_array = [[] for _ in 1:3]
    y_exact_end = y(t_e)
    for J in J_array
        S_1 = INTEGRATION_MATRIX_ARRAY_UNIFORM[8]
        S_2 = INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS[8]

        # IDC8-FE
        (t, y_out) = IDC_FE(ODE_system, 7, S_1, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[1], t[2] - t[1])
        push!(err_array[1], err)

        # IDC8-RK2-Heun's
        (t, y_out) = IDC_RK2_Heuns(ODE_system, 3, S_1, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[2], t[2] - t[1])
        push!(err_array[2], err)

        # IDC8-RK4
        (t, y_out) = IDC_RK4(ODE_system, 1, S_2, J, interpolation_polynomials((0:7)./7))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[3], t[2] - t[1])
        push!(err_array[3], err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    orders_to_plot = [8]
    log_const = [10]
    trunc_plot = [length(J_array)]
    for (i, order) in enumerate(orders_to_plot)
        order_Δt_array = Δt_array[i]
        err_order_array = log_const[i].*(order_Δt_array.^order)
        plot!(
            plot_err, order_Δt_array[1:trunc_plot[i]], err_order_array[1:trunc_plot[i]],
            linestyle = :dash, label = L"\sim(\Delta t)^{%$order}", color = :black
        )
    end
    data_color = [:red, :orange, :green]
    data_shape = [:circle, :hexagon, :square]
    data_label = ["IDC8-FE", "IDC8-RK2-Heun's", "IDC8-RK4"]
    for i in 1:3
        plot!(
            plot_err,
            Δt_array[i], err_array[i],
            markershape = data_shape[i], label = data_label[i], color = data_color[i],
            markersize = 3.5, markerstrokewidth = 0.8
        )
    end

    display(plot_err)
end

function test_IDC_levels()
    orders_to_plot = [2, 4, 8]

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    Δt_array = [[] for _ in 1:3]
    err_array = [[] for _ in 1:3]

    y_exact_end = y(t_e)
    N_array = 6:6:126
    for N in N_array
        # 1 CORRECTION
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[2]
        (t, y_out) = IDC_FE(ODE_system, 1, S, N)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[1], t[2] - t[1])
        push!(err_array[1], err)
    end
    for N in N_array
        # 3 CORRECTIONS
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[4]
        (t, y_out) = IDC_FE(ODE_system, 3, S, Int64(N/3))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[2], t[2] - t[1])
        push!(err_array[2], err)
    end
    # N_array = 5:5:120
    # for N in N_array
    #     # 5 CORRECTIONS
    #     S = INTEGRATION_MATRIX_ARRAY_UNIFORM[6]
    #     (t, y_out) = IDC_FE(ODE_system, 5, S, Int64(N/5))
    #     y_out_end = real(y_out[end, end])
    #     err = err_rel_norm(y_exact_end, y_out_end, 2)
    #     push!(Δt_array[3], t[2] - t[1])
    #     push!(err_array[3], err)
    # end
    N_array = 7:7:126
    for N in N_array
        # 7 CORRECTIONS
        S = INTEGRATION_MATRIX_ARRAY_UNIFORM[8]
        (t, y_out) = IDC_FE(ODE_system, 7, S, Int64(N/7))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array[3], t[2] - t[1])
        push!(err_array[3], err)

    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    colours = [:red, :orange, :blue]
    log_const = [1e-1, 3e-1, 3e-1]
    order_trunc = [size(Δt_array[1], 1), size(Δt_array[2], 1), size(N_array, 1) - 7]
    for (i, order) in enumerate(orders_to_plot)
        Δt_array_trunc = Δt_array[i][1:order_trunc[i]]
        err_order_array = log_const[i].*(Δt_array_trunc).^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array_trunc, err_order_array,
            linestyle = :dash, label = L"\sim(\Delta t)^{%$order}",
            color = colours[i]
        )
    end
    level_label = ["1 Correction", "3 Corrections", "7 Corrections"]
    for i in 1:3
        plot!(
            plot_err,
            Δt_array[i], err_array[i],
            markershape = :circle, label = level_label[i], color = colours[i]
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

function test_IDC_SDC_FE()
    number_corrections = 13
    N_array = number_corrections:number_corrections:number_corrections*20
    one_over_N_array = 1.0./N_array

    @unpack_ODETestSystem Jacobi_system
    @unpack_ODESystem ODE_system
    err_array = [[] for _ in 1:2]

    p = number_corrections + 1

    S_1 = INTEGRATION_MATRIX_ARRAY_UNIFORM[p]
    S_2 = INTEGRATION_MATRIX_ARRAY_LOBATTO[p]
    y_exact_end = y(t_e)
    for N in N_array
        # IDC-FE
        (t, y_out) = IDC_FE(ODE_system, number_corrections, S_1, Int64(N/(p - 1)))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[1], err)

        # SDC-FE with Lobatto nodes
        (t, y_out) = SDC_FE_lobatto(ODE_system, number_corrections, S_2, Int64(N/(p - 1)))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[2], err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 2.0
    )
    # order_plot_values = one_over_N_array.^p
    # plot!(
    #     plot_err,
    #     one_over_N_array, order_plot_values,
    #     linestyle = :dash, color = :black, label = L"N^{-%$p}\cdot 1"
    # )
    plot!(
        plot_err,
        one_over_N_array, err_array[1],
        markershape = :circle, label = string("IDC", p, "-FE"), color = :red,
    )
    plot!(
        plot_err,
        one_over_N_array, err_array[2],
        markershape = :circle, label = string("SDC", p, "-FE"), color = :green,
    )
    display(plot_err)
    ## PRINT ORDER
    for i in axes(err_array[2], 1)[2:end]
        order = (log(err_array[2][i]) - log(err_array[2][i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
        println(order)
    end    
end

function test_IDC_SDC_RK2()
    number_corrections = 5
    p = 2*(number_corrections + 1)
    N_array = (p - 1):(p - 1):15*(p - 1)
    one_over_N_array = 1.0./N_array

    @unpack_ODETestSystem Jacobi_system
    @unpack_ODESystem ODE_system

    S_1 = INTEGRATION_MATRIX_ARRAY_UNIFORM[p]
    S_2 = INTEGRATION_MATRIX_ARRAY_LOBATTO[p]

    err_array = [[] for _ in 1:2]
    y_exact_end = y(t_e)
    for N in N_array
        # IDCp-RK2-Heun's
        (t, y_out) = IDC_RK2_Heuns(ODE_system, 5, S_1, Int64(N/(p - 1)))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[1], err)


        # SDCp-RK2-Heun's with lobatto nodes
        (t, y_out) = SDC_RK2_Heuns_lobatto(ODE_system, 5, S_2, Int64(N/(p - 1)))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[2], err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        size = (800, 600), thickness_scaling = 1.0,
        legend = (1, 0),
    )
    orders_to_plot = p
    for order in orders_to_plot
        err_order_array = (one_over_N_array).^order
        plot!(
            plot_err, one_over_N_array, err_order_array,
            linestyle = :dash, label = L"1 \cdot N^{-%$order}"
        )
    end
    data_colours = [:orange, :purple]
    data_label = ["IDC8-RK2-Heun's", "SDC8-RK2-Heun's"]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            one_over_N_array, err_array[i],
            markershape = :hexagon, label = data_label[i], color = data_colours[i]
        )
    end
    # xtick_values = 1.0./[8, 16, 32, 64]
    # xtick_strings = ["1/8", "1/16", "1/32", "1/64"]
    # xticks!(plot_err, xtick_values, xtick_strings)

    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array[2], 1)[2:end]
        order = (log(err_array[2][i]) - log(err_array[2][i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
        println(order)
    end
end

function test_IDC_SDC_RK4()
    number_corrections = 3
    p = 4*(number_corrections + 1)
    N_array = (p - 1):(p - 1):15*(p - 1)
    one_over_N_array = 1.0./N_array

    @unpack_ODETestSystem Jacobi_system
    @unpack_ODESystem ODE_system

    S_1 = INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS[p]
    S_2 = INTEGRATION_MATRIX_ARRAY_LOBATTO_HALF_TIME_STEPS[p]

    err_array = [[] for _ in 1:2]
    y_exact_end = y(t_e)
    for N in N_array
        # IDCp-RK4-Heun's
        (t, y_out) = IDC_RK4(ODE_system, number_corrections, S_1, Int64(N/(p - 1)))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[1], err)


        # SDCp-RK4-Heun's with lobatto nodes
        (t, y_out) = SDC_RK4_lobatto(ODE_system, number_corrections, S_2, Int64(N/(p - 1)))
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[2], err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        size = (800, 600), thickness_scaling = 1.0,
        legend = (1, 0),
    )
    orders_to_plot = p
    for order in orders_to_plot
        err_order_array = (one_over_N_array).^order
        plot!(
            plot_err, one_over_N_array, err_order_array,
            linestyle = :dash, label = L"1 \cdot N^{-%$order}"
        )
    end
    data_colours = [:green, :purple]
    data_label = ["IDCp-RK4", "SDCp-RK4"]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            one_over_N_array, err_array[i],
            markershape = :square, label = data_label[i], color = data_colours[i]
        )
    end
    # xtick_values = 1.0./[8, 16, 32, 64]
    # xtick_strings = ["1/8", "1/16", "1/32", "1/64"]
    # xticks!(plot_err, xtick_values, xtick_strings)

    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array[2], 1)[2:end]
        order = (log(err_array[2][i]) - log(err_array[2][i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
        println(order)
    end
end

function test_SDC_RK2()
    J_array = 1:10

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    one_over_N_array = [[] for _ in 1:5]
    err_array = [[] for _ in 1:5]
    y_exact_end = y(t_e)
    for J in J_array
        # SDC-RK2-Heun's with lobatto nodes and 1 correction
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[4]
        (_, y_out) = SDC_RK2_Heuns_lobatto(ODE_system, 1, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(one_over_N_array[1], 1/(J*3))
        push!(err_array[1], err)

        # SDC-RK2-Heun's with lobatto nodes and 2 corrections
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[6]
        (_, y_out) = SDC_RK2_Heuns_lobatto(ODE_system, 2, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(one_over_N_array[2], 1/(J*5))
        push!(err_array[2], err)

        # SDC-RK2-Heun's with lobatto nodes and 3 corrections
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[8]
        (_, y_out) = SDC_RK2_Heuns_lobatto(ODE_system, 3, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(one_over_N_array[3], 1/(J*7))
        push!(err_array[3], err)

        # SDC-RK2-Heun's with lobatto nodes and 4 corrections
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[10]
        (_, y_out) = SDC_RK2_Heuns_lobatto(ODE_system, 4, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(one_over_N_array[4], 1/(J*9))
        push!(err_array[4], err)

        # SDC-RK2-Heun's with lobatto nodes and 5 corrections
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[12]
        (_, y_out) = SDC_RK2_Heuns_lobatto(ODE_system, 5, S, J)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(one_over_N_array[5], 1/(J*11))
        push!(err_array[5], err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        size = (800, 600), thickness_scaling = 1.0,
        legend = (1, 0),
    )
    # orders_to_plot = p
    # for order in orders_to_plot
    #     err_order_array = (one_over_N_array).^order
    #     plot!(
    #         plot_err, one_over_N_array, err_order_array,
    #         linestyle = :dash, label = L"1 \cdot N^{-%$order}"
    #     )
    # end
    data_colours = [:orange, :green, :blue, :purple, :red]
    data_label = ["1 correction", "2 corrections", "3 corrections", "4 corrections", "5 corrections"]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            one_over_N_array[i], err_array[i],
            markershape = :square, label = data_label[i], color = data_colours[i]
        )
    end
    # xtick_values = 1.0./[8, 16, 32, 64]
    # xtick_strings = ["1/8", "1/16", "1/32", "1/64"]
    # xticks!(plot_err, xtick_values, xtick_strings)

    display(plot_err)

    # ## PRINT ORDER
    for i in axes(err_array, 1)
        order = (log(err_array[i][2]) - log(err_array[i][1]))/(log(one_over_N_array[i][2]) - log(one_over_N_array[i][1]))
        println(order)
    end
end

function test_SDC_single()
    N_array = 5:40       # Number of time steps
    one_over_N_array = 1.0./N_array
    number_corrections = 1
    p = number_corrections + 1

    orders_to_plot = [p]

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    err_array = []
    y_exact_end = y(t_e)
    @showprogress "Testing SDC_single schemes..." for N in N_array
        # S = INTEGRATION_MATRIX_ARRAY_UNIFORM[N + 1]
        # S = INTEGRATION_MATRIX_ARRAY_LEGENDRE[N - 1]
        # S = INTEGRATION_MATRIX_ARRAY_LOBATTO[N + 1]
        S = integration_matrix(gausslobatto(N + 1)[1], gausslobatto(N + 1)[1]; use_double = false)
        # (t, y_out) = IDC_FE_single(ODE_system, number_corrections, S, N)
        # (t, y_out) = SDC_FE_legendre_single(ODE_system, number_corrections, S, N)
        (t, y_out) = SDC_FE_lobatto_single(ODE_system, number_corrections, S, N)
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array, err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        key = (1, 0), size = (800, 600), thickness_scaling = 4.0,
        legend = false,
    )
    for order in orders_to_plot
        err_order_array = 3.0.*(one_over_N_array).^order # Taking error constant = 1 always
        plot!(
            plot_err, one_over_N_array, err_order_array,
            linestyle = :dash, colour = :black#, label = L"1 \cdot N^{-%$order}"
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
    # one_over_N_array = 0.5.*[max((gausslobatto(N)[1][2:end] .- gausslobatto(N)[1][1:end - 1])...) for N in N_array]
    one_over_N_array = 1.0./N_array
    number_corrections_array = [1, 3, 5, 7]

    orders_to_plot = 1 .+ number_corrections_array

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    err_array = [[] for _ in axes(number_corrections_array, 1)]
    y_exact_end = y(t_e)
    for N in N_array
        # S = INTEGRATION_MATRIX_ARRAY_LEGENDRE[N - 1]
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[N + 1]
        # S = integration_matrix(gausslobatto(N + 1)[1], gausslobatto(N + 1)[1]; use_double = false)
        for (i, number_corrections) in enumerate(number_corrections_array)
            (_, y_out) = SDC_FE_lobatto_single(ODE_system, number_corrections, S, N)
            y_out_end = real(y_out[end, end])
            err = err_rel_norm(y_exact_end, y_out_end, 2)  # Relative 2-norm error
            push!(err_array[i], err)
        end
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        key = :bottomright, thickness_scaling = 4.0,
        legend = (1, 0), size = (1920, 1080)
    )
    colours = [:red, :orange, :green, :blue]
    log_const = [0.8e1, 0.5e1, 1e1, 1e2]
    order_trunc = [size(N_array, 1), size(N_array, 1) - 10, size(N_array, 1) - 15, 15]
    for (i, order) in enumerate(orders_to_plot)
        one_over_N_array_trunc = one_over_N_array[1:order_trunc[i]]
        err_order_array = log_const[i].*(one_over_N_array_trunc).^order # Taking error constant = 1 always
        plot!(
            plot_err, one_over_N_array_trunc, err_order_array,
            linestyle = :dash, label = L"\sim N^{-%$order}",
            color = colours[i]
        )
    end
    # cycle_index(index; max_index = 4) = (index - 1)%max_index + 1
    level_label = [string("Correction #", number_corrections) for number_corrections in number_corrections_array]
    for i in axes(number_corrections_array, 1)
        plot!(
            plot_err,
            one_over_N_array, err_array[i],
            markershape = :circle, label = level_label[i], color = colours[i]
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
    for i in axes(err_array[1], 1)[2:end]
        order = (log(err_array[3][i]) - log(err_array[3][i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
        println(order)
    end
end

function test_SDC()
    number_corrections_array = [3]
    N_array = 3:3:120
    one_over_N_array = 1.0./N_array

    @unpack_ODETestSystem poly_system_3
    @unpack_ODESystem ODE_system
    y_exact_end = y(t_e)

    p_array = []
    err_array = []
    for (i, number_corrections) in enumerate(number_corrections_array)
        p = number_corrections + 1
        M = p - 1
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[M + 1]
        push!(p_array, p)
        push!(err_array, [])
        for N in N_array
            (_, y_out) = SDC_FE_lobatto(ODE_system, number_corrections, S, Int64(N/M))
            y_out_end = real(y_out[end, end])
            err = err_rel_norm(y_exact_end, y_out_end, 2)
            push!(err_array[i], (err == 0.0) ? 1.0 : err)
        end

    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "1/N", ylabel = "||E||",
        key = (1, 0), size = (1000, 750), thickness_scaling = 4.0
    )
    data_colours = [:orange, :green, :blue, :purple]
    log_const = [1, 1, 1, 1]
    trunc_index = [size(N_array, 1), size(N_array, 1), size(N_array, 1), size(N_array, 1)]
    for (i, order) in enumerate(p_array)
        err_order_array = log_const[i].*one_over_N_array[1:trunc_index[i]].^order
        plot!(
            plot_err, one_over_N_array[1:trunc_index[i]], err_order_array,
            linestyle = :dash, label = L"N^{-%$order}\cdot %$(log_const[i])", color = data_colours[i]
        )
    end
    data_labels = [string("SDC", p, "-FE") for p in p_array]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            one_over_N_array, err_array[i],
            markershape = :circle, label = data_labels[i], color = data_colours[i]
        )
    end
    # xtick_values = 1.0./[18, 54, 108, 180]
    # xtick_strings = ["1/18", "1/54", "1/108", "1/180"]
    # xticks!(plot_err, xtick_values, xtick_strings)

    display(plot_err)

    for i in axes(err_array[1], 1)[2:end]
        order = (log(err_array[1][i]) - log(err_array[1][i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
        println(order)
    end
end

function test_SDC_reduced_stencil()
    number_corrections_array = [1, 3, 6, 9]
    p_array = 1 .+ number_corrections_array
    J_array = 1:20
    # one_over_N_array = [[] for _ in number_corrections_array]
    
    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system
    one_over_N_array = (t_e - t_s)./J_array
    y_exact_end = y(t_e)

    err_array = []
    for (i, number_corrections) in enumerate(number_corrections_array)
        M = ceil(Int64, (number_corrections + 1)/2)
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[M + 1]
        push!(err_array, [])
        for J in J_array
            (_, y_out) = SDC_FE_lobatto_reduced_stencil(ODE_system, number_corrections, S, J)
            # push!(one_over_N_array[i], 1/(size(y_out[:, end], 1) - 1))
            y_out_end = real(y_out[end, end])
            err = err_rel_norm(y_exact_end, y_out_end, 2)
            push!(err_array[i], (err == 0.0) ? 1.0 : err)
        end

    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), size = (1000, 750), thickness_scaling = 4.0
    )
    data_colours = [:orange, :green, :blue, :purple]
    log_const = [3e-1, 1e-3, 1e-7, 1e-12]
    trunc_index = [size(J_array, 1), size(J_array, 1), size(J_array, 1), 12]
    for (i, order) in enumerate(p_array)
        one_over_N_array_trunc = one_over_N_array[1:trunc_index[i]]
        err_order_array = log_const[i].*one_over_N_array_trunc.^order
        plot!(
            plot_err, one_over_N_array_trunc, err_order_array,
            linestyle = :dash, label = L"\sim N^{-%$order}", color = data_colours[i]
        )
    end
    data_labels = [string("SDC", p, "-FE") for p in p_array]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            one_over_N_array, err_array[i],
            markershape = :circle, label = data_labels[i], color = data_colours[i]
        )
    end
    xtick_values = [10^0.5, 1, 10^-0.5]
    # xtick_strings = ["1/12", "1/36", "1/96", "1/120"]
    xticks!(plot_err, xtick_values)

    display(plot_err)

    # for i in axes(err_array[1], 1)[2:end]
    #     order = (log(err_array[1][i]) - log(err_array[1][i - 1]))/(log(one_over_N_array[i]) - log(one_over_N_array[i - 1]))
    #     println(order)
    # end
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

