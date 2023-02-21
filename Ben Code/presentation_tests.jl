using
    Dates,
    Plots,
    LaTeXStrings,
    Measures,
    Colors,
    PyCall

include("ProjectTools.jl")

using .ProjectTools

const MATPLOTLIB = pyimport("matplotlib")
const RCPARAMS = PyDict(MATPLOTLIB["rcParams"])
RCPARAMS["mathtext.fontset"] = "cm"
# CMU_FONT = MATPLOTLIB["font_manager"][:FontProperties](fname="C:/Users/Ben/AppData/Local/Microsoft/Windows/Fonts/cmunrm.ttf")

pyplot()

function test_RIDC()
    N_array = [7, 10, 20, 40, 70]
    # J_array = [1, 2, 4, 8]
    p = 8
    M = p - 1
    S = integration_matrix_uniform(M)

    @unpack_ODETestSystem sqrt_system
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

function test_FE()
    N_array = [5, 10, 20, 40]
    orders_to_plot = 1

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system
    Δt_array = (t_e - t_s)./N_array
    err_array = []
    y_exact_end = y(t_e)
    for N in N_array
        (_, y_out) = RK1_forward_euler(ODE_system, N)
        y_out_end = real(y_out[end])
        err = err_rel(y_exact_end, y_out_end)
        push!(err_array, err)
    end

    RCPARAMS["xtick.major.pad"] = 10
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = raw"$Δt$", ylabel = raw"$||E||$",
        guidefontsize = 10, tickfontsize = 8, legendfontsize = 8,
        key = :bottomright, size = (800, 600), thickness_scaling = 3.5,
        xticks = [1e0, 10^(-0.5), 1e-1], yticks = [1e0, 10^(-0.5), 1e-1],
        xlims = (1e-1, 10^(0.05)), ylims = (1e-1, 1e0),
        legendposition = :topleft
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"||E|| = 1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

function test_IDC()
    J_array = [1, 2, 4, 8]
    orders_to_plot = 1:2

    M = 5
    S = integration_matrix_uniform(M)

    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system
    N_array = M.*J_array
    Δt_array = (t_e - t_s)./N_array
    err_array = []
    y_exact_end = y(t_e)
    for J in J_array
        (_, y_out) = IDC_FE(ODE_system, J, M, 1, S)
        y_out_end = real(y_out[end])
        err = err_rel(y_exact_end, y_out_end)
        push!(err_array, err)
    end

    RCPARAMS["xtick.major.pad"] = 10
    # RCPARAMS["mathtext.fontset"] = "cm"
    # plot_err = plot(
    #     xscale = :log10, yscale = :log10, xlabel = raw"huh $\Delta t$", ylabel = raw"wut $||E||$",
    #     tickfontsize = 10, legendfontsize = 5,
    #     key = :bottomright, size = (1000, 750), thickness_scaling = 3.0,
    #     # ylims = (10^(-3.2), 10^(-1.8)), yticks = [1e-3, 1e-2]
    # )
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = raw"$Δt$", ylabel = raw"$||E||$",
        guidefontsize = 10, tickfontsize = 8, legendfontsize = 8,
        key = :bottomright, size = (800, 600), thickness_scaling = 3.5,
        xticks = [1e0, 10^(-0.5), 1e-1], #yticks = [1e0, 10^(-0.5), 1e-1],
        xlims = (1e-1, 10^(0.05)),# ylims = (1e-1, 1e0),
        legendposition = :bottomright
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "", color = :blue,
    )
    for order in orders_to_plot
        err_order_array = 0.2Δt_array.^order # Taking error constant = 1 always
        label_string = (order == 1) ? "" : L"||E|| = 0.2\cdot (\Delta t)^%$order"
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = label_string
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

function display_FE_example()
    N_array = [5, 10, 20, 40]
    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    FE_out = []
    for N in N_array
        (t_out, η_out) = RK1_forward_euler(ODE_system, N)
        push!(FE_out, (t_out, real.(η_out)))
    end

    t_smooth = range(t_s, t_e, 1000)
    η_exact_solution_smooth = y.(t_smooth)

    time_plot = plot(
        size = (1050, 900), thickness_scaling = 7.0,
        guidefontsize = 10, tickfontsize = 8, legendfontsize = 8,
        xlabel = raw"$t$", ylabel = raw"$y(t)$",
        # xticks = [0.0, 0.5, 1.0], yticks = [0.4, 0.5, 0.6],
        # xlims = (-0.035, 1.035), ylims = (0.39, 0.625),
        xticks = [0, 5], yticks = [0, 600],
        # xlims = (-0.1, 5.1), ylims = (-100, 700),
        # legend = false,
        framestyle = :box,
        minorgrid = true,
        legendposition = :topleft
    )
    markershapes = [:circle, :square, :diamond, :hexagon]
    for (i, FE_out_tuple) in enumerate(reverse(FE_out))
        N = length(FE_out_tuple[1]) - 1
        plot!(
            time_plot, FE_out_tuple[1], FE_out_tuple[2],
            markershape = markershapes[i], markersize = 3, markerstrokewidth = 0.0,
            label = L"N = %$N"
        )
    end
    plot!(
        time_plot, t_smooth, η_exact_solution_smooth,
        linecolor = :black, label = ""
    )
end

function display_IDC_example()
    J_array = [1, 2, 4, 8]
    M = 5
    S = integration_matrix_uniform(M)
    @unpack_ODETestSystem sqrt_system
    @unpack_ODESystem ODE_system

    IDC_out = []
    for J in J_array
        (t_out, η_out) = IDC_FE(ODE_system, J, M, 1, S)
        push!(IDC_out, (t_out, real.(η_out)))
    end

    t_smooth = range(t_s, t_e, 1000)
    η_exact_solution_smooth = y.(t_smooth)

    # time_plot = plot(
    #     size = (1000, 750),
    #     guidefontfamily = "serif", guidefontsize = 30, tickfontfamily = "serif", tickfontsize = 20,
    #     xlabel = raw"$t$", ylabel = raw"$y(t)$",
    #     xticks = [0.8, 0.9, 1.0], yticks = [0.570, 0.572, 0.574],
    #     xlims = (0.79, 1.01), ylims = (0.5685, 0.575),
    #     # xticks = [0.0, 0.5, 1.0], yticks = [0.4, 0.5, 0.6],
    #     # xlims = (-0.035, 1.035), ylims = (0.375, 0.625),
    #     legend = false,
    #     framestyle = :box,
    # )
    time_plot = plot(
        size = (500, 600), thickness_scaling = 3.5,
        guidefontsize = 10, tickfontsize = 8,
        # yguidefontrotation = -90.0,
        xlabel = raw"$t$", ylabel = raw"$y(t)$",
        # xticks = [0.0, 0.5, 1.0], yticks = [0.4, 0.5, 0.6],
        # xlims = (-0.035, 1.035), ylims = (0.39, 0.625),
        xticks = [4.875, 5], yticks = [450, 550, 650],
        xlims = (4.8, 5.1), ylims = (450, 700),
        legend = false,
        framestyle = :box,
        minorgrid = true
    )
    markershapes = [:circle, :square, :diamond, :hexagon]
    for (i, IDC_out_tuple) in enumerate(reverse(IDC_out))
        plot!(
            time_plot, IDC_out_tuple[1], IDC_out_tuple[2],
            markershape = markershapes[i], markersize = 3, markerstrokewidth = 0.0
        )
    end
    plot!(
        time_plot, t_smooth, η_exact_solution_smooth,
        linecolor = :black
    )
end


function test_plot()
    font_plot = font(13, "serif")
    my_plot = plot(
        sin,
        size = (1000, 750),
        guidefontsize = 40, guidefont = "serif",
        xlabel = raw"$\Delta t_i$", ylabel = raw"$y$",
        tickfontsize = 30,
        linewidth = 5,
        label = raw"graph of $\sin(x)$",
        legend_font_pointsize = 16,
        annotations = (0, 0, "hi there"),
        # leftmargin = 10mm
    )
    plot!(
        my_plot, (0.5π, 1.0),
        markershape = :circle, markercolor = :red, markersize = 10.0,
        markerstrokewidth = 0.0,
        label = ""
    )
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/$dtstring-test_plot.png"
    savefig(my_plot, fname)
    display(my_plot)
end