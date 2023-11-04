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
    orders_to_plot = [1]

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    N_array = 10:10:60
    Δt_array = []
    err_array = []

    y_exact_end = y(t_e)
    for N in N_array
        (t, y_out) = RK1_forward_euler(ODE_system, N)
        y_out_end = real(y_out[end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(Δt_array, t[2] - t[1])
        push!(err_array, (err  == 0.0) ? 1.0 : err)
    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "Δt", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    order_colour = [:black]
    for (i, order) in enumerate(orders_to_plot)
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta t)^{%$order}", color = order_colour[i]
        )
    end
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "FE", color = :blue,
    )
    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array, 1)[2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(Δt_array[i]) - log(Δt_array[i - 1]))
        println(order)
    end
end

function test_multiple_RK_methods()
    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    N_array = 10:10:60
    # one_over_N_array = 1.0./N_array

    methods_array = [RK2_Heuns, RK2_midpoint, RK3_Kutta, RK4_standard]
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
    orders_to_plot = [2, 3, 4]
    coeffs = [1e-1, 10^-2, 10^-2.3]
    order_trunc = [size(N_array, 1), size(N_array, 1), size(N_array, 1)]
    order_colours = [:grey, :green, :blue]
    for (i, order) in enumerate(orders_to_plot)
        Δt_array_trunc = Δt_array[i][1:order_trunc[i]]
        err_order_array = coeffs[i].*Δt_array_trunc.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array_trunc, err_order_array,
            linestyle = :dash, label = L"\sim(\Delta t)^{%$order}",
            color = order_colours[i]
        )
    end
    method_colours = [:red, :orange, :green, :blue]
    method_labels = ["Heun's RK2", "Midpoint RK2", "Kutta's RK3", "Standard RK4"]
    method_markers = [:square, :diamond, :dtriangle, :hexagon]
    for (i, err) in enumerate(err_array)
        plot!(
            plot_err,
            Δt_array[i], err,
            markershape = method_markers[i], label = method_labels[i], color = method_colours[i]
        )
    end
    # xtick_values = [10^-0.5, 10^-1.0, 10^-1.5]
    # xtick_strings = ["1/5", "1/10", "1/20", "1/30", "1/40"]
    # xticks!(plot_err, xtick_values)
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
    number_corrections_array = [5]
    N_array = 3:3:120
    one_over_N_array = 1.0./N_array

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system
    y_exact_end = y(t_e)

    p_array = []
    err_array = []
    for (i, number_corrections) in enumerate(number_corrections_array)
        p = number_corrections + 1
        # M = p - 1
        M = ceil(Int64, (number_corrections + 1)/2)
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[M + 1]
        push!(p_array, p)
        push!(err_array, [])
        for N in N_array
            (_, y_out) = SDC_FE_lobatto_reduced_quadrature(ODE_system, number_corrections, S, Int64(N/M))
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
    
    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system
    ΔT_array = (t_e - t_s)./J_array
    y_exact_end = y(t_e)

    err_array = []
    for (i, number_corrections) in enumerate(number_corrections_array)
        M = ceil(Int64, (number_corrections + 1)/2)
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[M + 1]
        push!(err_array, [])
        for J in J_array
            (_, y_out) = SDC_FE_lobatto_reduced_quadrature(ODE_system, number_corrections, S, J)
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
        ΔT_array_trunc = ΔT_array[1:trunc_index[i]]
        err_order_array = log_const[i].*ΔT_array_trunc.^order
        plot!(
            plot_err, ΔT_array_trunc, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^{%$order}", color = data_colours[i]
        )
    end
    data_labels = [string("SDC", p, "-FE") for p in p_array]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            ΔT_array, err_array[i],
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

function test_SDC_RK2_reduced_stencil()
    number_corrections_array = [1, 2, 3, 4]
    p_array = 2 .* (1 .+ number_corrections_array)
    orders_to_plot = [2, 3, 4, 5]
    J_array = 1:20
    
    @unpack_ODETestSystem exp_system_3
    @unpack_ODESystem ODE_system
    ΔT_array = (t_e - t_s)./J_array
    y_exact_end = y(t_e)

    err_array = []
    for (i, number_corrections) in enumerate(number_corrections_array)
        M = number_corrections + 1
        S = INTEGRATION_MATRIX_ARRAY_LOBATTO[M + 1]
        push!(err_array, [])
        for J in J_array
            (_, y_out) = SDC_RK2_Heuns_lobatto_reduced_quadrature_convergence_enhanced(ODE_system, number_corrections, S, J, 0)
            y_out_end = real(y_out[end, end])
            err = err_rel_norm(y_exact_end, y_out_end, 2)
            push!(err_array[i], (err == 0.0) ? 1.0 : err)
        end

    end

    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), size = (1000, 750), thickness_scaling = 4.0
    )
    data_colours = [:red, :orange, :green, :blue]
    log_const = [1, 1, 1, 1]
    trunc_index = [size(J_array, 1), size(J_array, 1), size(J_array, 1), 12]
    for (i, order) in enumerate(orders_to_plot)
        ΔT_array_trunc = ΔT_array[1:trunc_index[i]]
        err_order_array = log_const[i].*ΔT_array_trunc.^order
        plot!(
            plot_err, ΔT_array_trunc, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^{%$order}", color = data_colours[i]
        )
    end
    data_labels = [string("SDC", p, "-RK2-Heun's") for p in p_array]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            ΔT_array, err_array[i],
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

function test_PCIDC_FE_reduced_stencil()
    J_array = 1:20
    number_corrections = 8
    number_final_level_intervals = 1
    orders_to_plot = 1:(number_corrections .+ 1)
    
    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system
    
    M_levels = [level for level in 1:number_corrections]
    push!(M_levels, number_final_level_intervals)
    uniform_nodes_levels = [range(0, 1, M + 1) for M in M_levels]
    S_levels = [integration_matrix(uniform_nodes_levels[level - 1], uniform_nodes_levels[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases = [interpolation_polynomials(uniform_nodes_levels[level]) for level in 1:number_corrections]


    ΔT_array = (t_e - t_s)./J_array
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (_, y_out) = parallel_IDC_FE_reduced_stencil(
            ODE_system,
            number_corrections, S_levels, J, number_final_level_intervals, lagrange_bases
        )
        y_out_end = real(y_out[end][end])
        err = err_rel(y_exact_end, y_out_end)
        push!(err_array, err)
    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), size = (1000, 750)
    )
    plot!(
        plot_err,
        ΔT_array, err_array,
        markershape = :circle, label = "", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = ΔT_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, ΔT_array, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array)[1][2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(ΔT_array[i]) - log(ΔT_array[i - 1]))
        println(order)
    end
end

function test_PCSDC_FE_lobatto_reduced_stencil()
    J_array = 1:20
    number_corrections = 7
    number_final_level_intervals = 1
    orders_to_plot = 1:(number_corrections .+ 1)

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    M_levels = [level for level in 1:number_corrections]
    # M_levels = [ceil(Int64, (level + 1)/2) for level in 1:number_corrections]
    push!(M_levels, number_final_level_intervals)
    lobatto_nodes_levels = [gausslobatto(M + 1)[1] for M in M_levels]
    S_levels = [integration_matrix(lobatto_nodes_levels[level - 1], lobatto_nodes_levels[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases = [interpolation_polynomials((gausslobatto(1 + M_levels[level - 1])[1] .+ 1)./2) for level in 2:(number_corrections + 1)]


    ΔT_array = (t_e - t_s)./J_array
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (_, y_out) = PCSDC_FE_lobatto_reduced_stencil(
            ODE_system,
            number_corrections, S_levels, J, number_final_level_intervals, lagrange_bases
        )
        y_out_end = real(y_out[end][end])
        err = err_rel(y_exact_end, y_out_end)
        push!(err_array, err)
    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), size = (1000, 750)
    )
    plot!(
        plot_err,
        ΔT_array, err_array,
        markershape = :circle, label = "", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = ΔT_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, ΔT_array, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)

    ## PRINT ORDER
    for i in axes(err_array)[1][2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(ΔT_array[i]) - log(ΔT_array[i - 1]))
        println(order)
    end
end

function test_PCIDC()
    number_corrections = 5
    J_array = number_corrections:20
    # M = number_corrections
    M = ceil(Int64, (number_corrections + 1)/2)
    p = number_corrections + 1
    # p = 2*(number_corrections + 1)
    orders_to_plot = [p - 1, p]

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    S = INTEGRATION_MATRIX_ARRAY_LOBATTO[M + 1]
    # S = INTEGRATION_MATRIX_ARRAY_UNIFORM[M + 1]
    ΔT_array = (t_e - t_s)./J_array
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (_, y_out) = PCSDC_FE_lobatto_reduced_quadrature(
            ODE_system,
            number_corrections, S, J
        )
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array, err)
    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    for order in orders_to_plot
        err_order_array = ΔT_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, ΔT_array, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^%$order",
            color = :black
        )
    end
    plot!(
        plot_err,
        ΔT_array, err_array,
        markershape = :circle, label = "", color = :blue,
    )

    # PRINT ORDER
    for i in axes(err_array, 1)[2:end]
        order = (log(err_array[i]) - log(err_array[i - 1]))/(log(ΔT_array[i]) - log(ΔT_array[i - 1]))
        println(order)
    end

    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

function test_PCIDC4()
    J_array = 1:20
    # M = ceil(Int64, (number_corrections + 1)/2)
    # p = 2*(number_corrections + 1)
    orders_to_plot = 4

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    number_corrections = 3
    M_1 = number_corrections
    M_2 = ceil(Int64, (number_corrections + 1)/2)
    S_1 = INTEGRATION_MATRIX_ARRAY_UNIFORM[M_1 + 1]
    S_2 = INTEGRATION_MATRIX_ARRAY_LOBATTO[M_2 + 1]
    ΔT_array = (t_e - t_s)./J_array
    err_array = [[] for _ in 1:4]
    y_exact_end = y(t_e)
    for J in J_array
        # PCIDC4-FE
        (_, y_out) = PCIDC_FE(
            ODE_system,
            3, S_1, J
        )
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[1], err)

        # PCSDC4-FE
        (_, y_out) = PCSDC_FE_lobatto_reduced_quadrature(
            ODE_system,
            3, S_2, J
        )
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[2], err)

        # PCIDC4-RK2
        (_, y_out) = PCIDC_RK2_Heuns(
            ODE_system,
            1, S_1, J
        )
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[3], err)

        # IDC4-FE
        (_, y_out) = IDC_FE(
            ODE_system,
            3, S_1, J
        )
        y_out_end = real(y_out[end, end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[4], err)
    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    for order in orders_to_plot
        err_order_array = (1e-4).*ΔT_array.^order
        plot!(
            plot_err, ΔT_array, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^%$order",
            color = :darkgreen
        )
    end
    data_colours = [:red, :orange, :green, :blue]
    data_labels = ["PCIDC4-FE", "PCSDC4-FE", "PCIDC4-RK2", "IDC4-FE"]
    data_markers = [:circle, :circle, :square, :circle]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            ΔT_array, err_array[i],
            markershape = data_markers[i], label = data_labels[i], color = data_colours[i]
        )
    end
    plot_ylims = ylims(plot_err)
    plot!(
        plot_err,
        [(ΔT_array[2], plot_ylims[1]), (ΔT_array[2], plot_ylims[2])],
        linestyle = :dot, label = "", color = :black, order = :back
    )
    plot!(
        plot_err,
        [(ΔT_array[4], plot_ylims[1]), (ΔT_array[4], plot_ylims[2])],
        linestyle = :dot, label = "", color = :black, order = :back
    )

    xtick_values = [10^0.5, 1, 10^-0.5]
    # xtick_strings = ["1/12", "1/36", "1/96", "1/120"]
    xticks!(plot_err, xtick_values)


    display(plot_err)
end

function test_PCIDC_reduced_stencils()
    J_array = 1:20
    number_corrections = 7
    p = 1*(number_corrections + 1)
    orders_to_plot = [p - 1, p]

    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    # S = integration_matrix_uniform_RK4(p - 1)
    number_final_level_intervals = 3
    t_levels = [gausslobatto(l + 1)[1] for l in 1:number_corrections]
    push!(t_levels, gausslobatto(number_final_level_intervals + 1)[1])
    S_levels = [integration_matrix(t_levels[l], t_levels[l + 1]) for l in 1:number_corrections]
    ΔT_array = (t_e - t_s)./J_array
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (_, y_out) = parallel_SDC_FE_lobatto_reduced_stencil(
            ODE_system,
            number_corrections, S_levels, J, number_final_level_intervals
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array, err)
    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), size = (1000, 750), thickness_scaling = 4.0
    )
    plot!(
        plot_err,
        ΔT_array, err_array,
        markershape = :circle, label = "", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = ΔT_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, ΔT_array, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^%$order"
        )
    end

    display(plot_err)
end

function test_PCIDC6_reduced_stencils()
    J_array = 1:20
    orders_to_plot = [6]
    number_corrections = 5

    number_final_level_intervals_one = 1
    number_final_level_intervals_usual_IDC = 6
    number_final_level_intervals_usual_SDC = 4
    number_final_level_intervals_big = 20

    M_levels_1 = [level for level in 1:number_corrections]
    push!(M_levels_1, number_final_level_intervals_one)
    uniform_nodes_levels_1 = [range(0, 1, M + 1) for M in M_levels_1]
    S_levels_1 = [integration_matrix(uniform_nodes_levels_1[level - 1], uniform_nodes_levels_1[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_1 = [interpolation_polynomials(uniform_nodes) for uniform_nodes in uniform_nodes_levels_1]

    M_levels_2 = [level for level in 1:number_corrections]
    push!(M_levels_2, number_final_level_intervals_usual_IDC)
    uniform_nodes_levels_2 = [range(0, 1, M + 1) for M in M_levels_2]
    S_levels_2 = [integration_matrix(uniform_nodes_levels_2[level - 1], uniform_nodes_levels_2[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_2 = [interpolation_polynomials(uniform_nodes) for uniform_nodes in uniform_nodes_levels_2]

    M_levels_3 = [level for level in 1:number_corrections]
    push!(M_levels_3, number_final_level_intervals_big)
    uniform_nodes_levels_3 = [range(0, 1, M + 1) for M in M_levels_3]
    S_levels_3 = [integration_matrix(uniform_nodes_levels_3[level - 1], uniform_nodes_levels_3[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_3 = [interpolation_polynomials(uniform_nodes) for uniform_nodes in uniform_nodes_levels_3]

    M_levels_4 = [ceil(Int64, (level + 1)/2) for level in 1:number_corrections]
    push!(M_levels_4, number_final_level_intervals_one)
    lobatto_nodes_levels_4 = [gausslobatto(M + 1)[1] for M in M_levels_4]
    S_levels_4 = [integration_matrix(lobatto_nodes_levels_4[level - 1], lobatto_nodes_levels_4[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_4 = [interpolation_polynomials((gausslobatto(1 + M_levels_4[level - 1])[1] .+ 1)./2) for level in 2:(number_corrections + 1)]

    M_levels_5 = [ceil(Int64, (level + 1)/2) for level in 1:number_corrections]
    push!(M_levels_5, number_final_level_intervals_usual_SDC)
    lobatto_nodes_levels_5 = [gausslobatto(M + 1)[1] for M in M_levels_5]
    S_levels_5 = [integration_matrix(lobatto_nodes_levels_5[level - 1], lobatto_nodes_levels_5[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_5 = [interpolation_polynomials((gausslobatto(1 + M_levels_5[level - 1])[1] .+ 1)./2) for level in 2:(number_corrections + 1)]

    M_levels_6 = [ceil(Int64, (level + 1)/2) for level in 1:number_corrections]
    push!(M_levels_6, number_final_level_intervals_big)
    lobatto_nodes_levels_6 = [gausslobatto(M + 1)[1] for M in M_levels_6]
    S_levels_6 = [integration_matrix(lobatto_nodes_levels_6[level - 1], lobatto_nodes_levels_6[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_6 = [interpolation_polynomials((gausslobatto(1 + M_levels_6[level - 1])[1] .+ 1)./2) for level in 2:(number_corrections + 1)]


    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    ΔT_array = (t_e - t_s)./J_array
    err_array = [[] for _ in 1:6]
    y_exact_end = y(t_e)
    for J in J_array
        # PCIDC6-FE, L = 1
        (_, y_out) = PCIDC_FE_reduced_stencil(
            ODE_system,
            number_corrections, S_levels_1, J, number_final_level_intervals_one, lagrange_bases_1
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[1], err)

        # PCIDC6-FE, L = 6
        (_, y_out) = PCIDC_FE_reduced_stencil(
            ODE_system,
            number_corrections, S_levels_2, J, number_final_level_intervals_usual_IDC, lagrange_bases_2
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[2], err)

        # PCIDC6-FE, L = 20
        (_, y_out) = PCIDC_FE_reduced_stencil(
            ODE_system,
            number_corrections, S_levels_3, J, number_final_level_intervals_big, lagrange_bases_3
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[3], err)

        # PCSDC6-FE, L = 1
        (_, y_out) = PCSDC_FE_lobatto_reduced_stencil_and_quadrature(
            ODE_system,
            number_corrections, S_levels_4, J, number_final_level_intervals_one, lagrange_bases_4
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[4], err)

        # PCSDC6-FE, L = 4
        (_, y_out) = PCSDC_FE_lobatto_reduced_stencil_and_quadrature(
            ODE_system,
            number_corrections, S_levels_5, J, number_final_level_intervals_usual_SDC, lagrange_bases_5
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[5], err)

        # PCSDC6-FE, L = 20
        (_, y_out) = PCSDC_FE_lobatto_reduced_stencil_and_quadrature(
            ODE_system,
            number_corrections, S_levels_6, J, number_final_level_intervals_big, lagrange_bases_6
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[6], err)

    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    order_colors = [:black]
    order_const = [10^-5.5]
    for (i, order) in enumerate(orders_to_plot)
        err_order_array = order_const[i].*ΔT_array.^order
        plot!(
            plot_err, ΔT_array, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^%$order",
            color = order_colors[i]
        )
    end
    data_colours = [:red, :orange, :lime, :green, :blue, :purple]
    data_labels = ["PCIDC6-FE, L = 1", "PCIDC6-FE, L = 6", "PCIDC6-FE, L = 20", "PCSDC6-FE, L = 1", "PCSDC6-FE, L = 4", "PCSDC6-FE, L = 20"]
    data_markers = [:circle, :circle, :circle, :diamond, :diamond, :diamond]
    for i in axes(err_array, 1)[1:3]
        plot!(
            plot_err,
            ΔT_array, err_array[i],
            markershape = data_markers[i], label = data_labels[i], color = data_colours[i]
        )
    end
    plot_ylims = ylims(plot_err)
    plot!(
        plot_err,
        [(ΔT_array[6], plot_ylims[1]), (ΔT_array[6], plot_ylims[2])],
        linestyle = :dot, label = "", color = :black
    )

    # xtick_values = [10^0.5, 1, 10^-0.5]
    # xtick_strings = ["1/12", "1/36", "1/96", "1/120"]
    # xticks!(plot_err, xtick_values)


    # PRINT ORDER
    for e in axes(err_array, 1)
        println()
        println("e = ", e)
        for i in axes(err_array[e], 1)[2:end]
            order = (log(err_array[e][i]) - log(err_array[e][i - 1]))/(log(ΔT_array[i]) - log(ΔT_array[i - 1]))
            println(order)
        end
    end

    display(plot_err)
end

function test_PCSDC6_reduced_stencils_increased_quadrature()
    J_array = 1:20
    number_corrections = 7
    orders_to_plot = [number_corrections + 1]

    number_final_level_intervals_1 = 1
    number_final_level_intervals_2 = 6
    number_final_level_intervals_3 = 20
    
    M_levels_1 = [level for level in 1:number_corrections]
    push!(M_levels_1, number_final_level_intervals_1)
    lobatto_nodes_levels_1 = [gausslobatto(M + 1)[1] for M in M_levels_1]
    S_levels_1 = [integration_matrix(lobatto_nodes_levels_1[level - 1], lobatto_nodes_levels_1[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_1 = [interpolation_polynomials((gausslobatto(1 + M_levels_1[level - 1])[1] .+ 1)./2) for level in 2:(number_corrections + 1)]

    M_levels_2 = [level for level in 1:number_corrections]
    push!(M_levels_2, number_final_level_intervals_2)
    lobatto_nodes_levels_2 = [gausslobatto(M + 1)[1] for M in M_levels_2]
    S_levels_2 = [integration_matrix(lobatto_nodes_levels_2[level - 1], lobatto_nodes_levels_2[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_2 = [interpolation_polynomials((gausslobatto(1 + M_levels_2[level - 1])[1] .+ 1)./2) for level in 2:(number_corrections + 1)]

    M_levels_3 = [level for level in 1:number_corrections]
    push!(M_levels_3, number_final_level_intervals_3)
    lobatto_nodes_levels_3 = [gausslobatto(M + 1)[1] for M in M_levels_3]
    S_levels_3 = [integration_matrix(lobatto_nodes_levels_3[level - 1], lobatto_nodes_levels_3[level]) for level in 2:(number_corrections + 1)]
    lagrange_bases_3 = [interpolation_polynomials((gausslobatto(1 + M_levels_3[level - 1])[1] .+ 1)./2) for level in 2:(number_corrections + 1)]


    @unpack_ODETestSystem log_system
    @unpack_ODESystem ODE_system

    ΔT_array = (t_e - t_s)./J_array
    err_array = [[] for _ in 1:3]
    y_exact_end = y(t_e)
    for J in J_array
        # PCSDC6-FE, L = 1
        (_, y_out) = PCSDC_FE_lobatto_reduced_stencil(
            ODE_system,
            number_corrections, S_levels_1, J, number_final_level_intervals_1, lagrange_bases_1
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[1], err)

        # PCSDC6-FE, L = 6
        (_, y_out) = PCSDC_FE_lobatto_reduced_stencil(
            ODE_system,
            number_corrections, S_levels_2, J, number_final_level_intervals_2, lagrange_bases_2
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[2], err)

        # PCSDC6-FE, L = 20
        (_, y_out) = PCSDC_FE_lobatto_reduced_stencil(
            ODE_system,
            number_corrections, S_levels_3, J, number_final_level_intervals_3, lagrange_bases_3
        )
        y_out_end = real(y_out[end][end])
        err = err_rel_norm(y_exact_end, y_out_end, 2)
        push!(err_array[3], err)

    end

    ## MAKE ORDER PLOT
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = "ΔT", ylabel = "||E||",
        key = (1, 0), thickness_scaling = 4.0
    )
    order_colors = [:black]
    order_const = [10^-6.5]
    for (i, order) in enumerate(orders_to_plot)
        err_order_array = order_const[i].*ΔT_array.^order
        plot!(
            plot_err, ΔT_array, err_order_array,
            linestyle = :dash, label = L"\sim (\Delta T~)^%$order",
            color = order_colors[i]
        )
    end
    data_colours = [:green, :blue, :purple]
    data_labels = [
        string("PCSDC6-FE, L = ", number_final_level_intervals_1),
        string("PCSDC6-FE, L = ", number_final_level_intervals_2),
        string("PCSDC6-FE, L = ", number_final_level_intervals_3)
    ]
    data_markers = [:diamond, :diamond, :diamond]
    for i in axes(err_array, 1)
        plot!(
            plot_err,
            ΔT_array, err_array[i],
            markershape = data_markers[i], label = data_labels[i], color = data_colours[i]
        )
    end
    plot_ylims = ylims(plot_err)
    plot!(
        plot_err,
        [(ΔT_array[6], plot_ylims[1]), (ΔT_array[6], plot_ylims[2])],
        linestyle = :dot, label = "", color = :black
    )

    # xtick_values = [10^0.5, 1, 10^-0.5]
    # xtick_strings = ["1/12", "1/36", "1/96", "1/120"]
    # xticks!(plot_err, xtick_values)


    # PRINT ORDER
    for e in axes(err_array, 1)
        println()
        println("e = ", e)
        for i in axes(err_array[e], 1)[2:end]
            order = (log(err_array[e][i]) - log(err_array[e][i - 1]))/(log(ΔT_array[i]) - log(ΔT_array[i - 1]))
            println(order)
        end
    end

    display(plot_err)
end

