using Dates, Plots, BenchmarkTools

""" Simple bump function """
bump_init(x; A=1.0, d=0.0, s=1.0) = A*exp(-((x+d)/s)^2)

""" Perform forward Euler simulation on advection equation """
function advection_forward_euler(
    x_start, x_end, t_end, v, make_animation=true;
    x_res=250, t_res=500, max_frames=100
)
    # Set Parameters
    Δx = (x_end - x_start) / x_res
    Δt = t_end / t_res
    k = v*Δt/Δx

    x_array = range(x_start, x_end, x_res) |> collect
    t_array = range(0.0, t_end, t_res) |> collect
    u_array = bump_init.(x_array)
    u_arrays = [u_array]

    # Trot on
    for t in t_array[2:end]
        u_new = u_array[2:end] - k*(u_array[2:end] - u_array[1:end-1])
        u_new = vcat([0], u_new)
        u_array = u_new
        push!(u_arrays, u_array)
    end

    # Animate simulation alongside accurate solution
    if make_animation
        u_corrects = [bump_init.(x_array .- v*t) for t in t_array]
        max_frames = (max_frames > t_res) ? t_res : max_frames # Ensures at least 1 frame

        anim = @animate for (i, u) in enumerate(u_arrays)
            frame_plot = plot(x_array, u)
            plot!(frame_plot, x_array, u_corrects[i], color=:red)
        end every fld(t_res, 100)

        dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
        fname = "Ben Code/animation-$dtstring.mov"
        mov(anim, fname, fps=20)
    end
end

function main()
    test_x_start = -2.0
    test_x_end = 8.0
    test_t_end = 8.0
    test_v = 1.0
    advection_forward_euler(test_x_start, test_x_end, test_t_end, test_v, true)
    # @benchmark advection_forward_euler(-2.0, 8.0, 8.0, 1.0, false) # Can't put in variables??
end