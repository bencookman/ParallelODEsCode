using
    Dates,
    Plots,
    LaTeXStrings

include("ProjectTools.jl")

using .ProjectTools

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

function test_FE()
    N_array = 10:3:52

    @unpack_ODETestSystem Butcher_p53_system
    @unpack_ODESystem ODE_system

    orders_to_plot = 1
    Δt_array = (t_e - t_s)./N_array
    err_array = []
    y_exact_end = y(t_e)
    for N in N_array
        (_, y_out) = RK1_forward_euler(ODE_system, N)
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
        markershape=:circle, label="Solution approximated with forward Euler", color = :blue,
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
