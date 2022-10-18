using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra, LaTeXStrings, GR

include("ProjectTools.jl")

using .ProjectTools

function butcher1_forward_euler(t_end, t_res, u_init; save_plot=true, save_type="png")
    t_array = range(0.0, t_end, t_res+1) |> collect
    Δt = t_end/t_res
    f(t, u) = (u-2t*u^2)/(1+t)
    u_correct(t) = (1+t)/(t^2+1/u_init)

    u = u_init
    u_sims = [u]
    for t in t_array[1:end-1]
        u_new = u + Δt*f(t, u)
        u = u_new
        push!(u_sims, u)
    end

    if save_plot
        new_plot = plot(
            t_array, u_sims,
            label="h=$(Δt)", marker=".",
            xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0.2, 0.4, 0.6, 0.8]
        )
        t_plot = range(0.0, t_end, 1000) |> collect
        plot!(
            new_plot, t_plot, u_correct.(t_plot),
            color=:red, ylims=(0.2, 0.8), label="u_correct"
        )
        display(new_plot)

        dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
        fname = "Ben Code/output/plot-$dtstring.$(save_type)" # NEED TO IMPROVE
        savefig(new_plot, fname)
    end

    # Calculate error on whole series
    error_calculate(u_correct.(t_array), u_sims, 2)
end

function run_butcher1()
    t_end = 1.0
    u_init = 0.4
    t_res_array = [5*2^i for i in 0:20]
    err_plot_array = []

    for t_res in t_res_array
        err = butcher1_forward_euler(t_end, t_res, u_init; save_plot=false)
        # println("res = $(t_res)")
        # println("error norm = $(err[:norm])")
        # println("error at end = $(err[:data][end])")
        push!(err_plot_array, err[:norm])
    end

    plt = plot(
        t_res_array, err_plot_array,
        xscale=:log10, yscale=:log10, xlabel=L"t_{res}", ylabel=L"l^2" * " error",
        title="test title 1"
    )
end
