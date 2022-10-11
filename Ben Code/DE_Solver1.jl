using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra, LaTeXStrings, GR

function error_calculate(data_correct, data_simulated, p)
    err_data = abs.(data_correct - data_simulated)
    err_norm = norm(err_data, p)
    err_max  = maximum(err_data)
    return Dict(:data => err_data, :norm => err_norm, :max => err_max)
end

""" Simple bump function """
bump_init(x; A=1.0, d=0.0, s=1.0) = A*exp(-((x+d)/s)^2)

"""
Perform forward Euler simulation on advection equation.
Can generate .mov file of animation and return error information when specified.
"""
function advection_forward_euler(
    x_start, x_end, t_end, v, make_animation, return_error;
    x_res=250, t_res=500, max_frames=100
)
    # Set Parameters
    Δx = (x_end - x_start)/x_res
    Δt = t_end/t_res
    k = v*Δt/Δx
    x_array = range(x_start, x_end, x_res) |> collect
    t_array = range(0.0, t_end, t_res) |> collect
    u_array = bump_init.(x_array)

    # Trot on
    u_arrays = [u_array]
    for t in t_array[2:end]
        u_new = u_array[2:end] - k*(u_array[2:end] - u_array[1:end-1])
        u_new = vcat([0], u_new)
        u_array = u_new
        push!(u_arrays, u_array)
    end

    # We know solutions to the advection equation follow paths u(x, t) = g(x+vt) + f(x-vt)
    global u_corrects
    if make_animation || return_error
        u_corrects = [bump_init.(x_array .- v*t) for t in t_array]
    end

    # Animate simulation alongside accurate solution
    if make_animation
        max_frames = (max_frames > t_res) ? t_res : max_frames # Ensures at least 1 frame
        anim = @animate for (i, u) in enumerate(u_arrays)
            frame_plot = plot(x_array, u)
            plot!(frame_plot, x_array, u_corrects[i], color=:red)
        end every fld(t_res, 100)

        dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
        fname = "Ben Code/output/animation-$dtstring.mov" # NEED TO IMPROVE
        mov(anim, fname, fps=20)
    end

    # Error calculation (calculated at t=end)
    return_error && error_calculate(u_corrects[end], u_arrays[end], 2)
end

function run_advection_forward_euler()
    test_x_start = -2.0
    test_x_end = 8.0
    test_t_end = 8.0
    test_v = 1.0
    err = advection_forward_euler(test_x_start, test_x_end, test_t_end, test_v, false, true)
    println("error l-2 norm: $(err[:norm])")
    println("error max: $(err[:max])")
    # When benchmarking, must make sure to 'interpolate constant values in the benchmarking context'
    # https://juliaci.github.io/BenchmarkTools.jl/stable/manual/#Interpolating-values-into-benchmark-expressions
    # This took me like an hour to figure out (gives error in main function otherwise)
    # @benchmark advection_forward_euler($(test_x_start), $(test_x_end), $(test_t_end), $(test_v), false, false)
end

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
