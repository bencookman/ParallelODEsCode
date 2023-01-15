using FFTW, Dates, ProgressMeter, LaTeXStrings, Plots

include("ProjectTools.jl")

using .ProjectTools

u_SOLITON_EXACT(t, x; k = 1.0, x_0 = 0.0) = 12*k^2*(sech(k*(x - x_0) - 4*k^3*t))^2

u_1_EXACT(t, x) = u_SOLITON_EXACT(t, x; k = 0.3, x_0 = -20.0)   # On (-15π, 15π)
u_1_INITIAL(x) = u_1_EXACT(0, x)
u_2_EXACT(t, x) = u_SOLITON_EXACT(t, x; k = 0.2, x_0 = 0.0)     # On (-15π, 15π)
u_2_INITIAL(x) = u_1_EXACT(0, x)

""" Solve the KdV equation using spectral methods on the region (0, 2π). """
function KdV_solver_0_to_2pi(;
    u_initial::Function = u_1_INITIAL,
    t_e::Float64 = 1.0,
    sf::Float64 = 1.0,
    N_t::Int64 = 10_000,
    N_x::Int64 = 2^8
)
    t = range(0.0, t_e, N_t)
    Δt = t[2] - t[1]
    Δx = 2π/N_x
    x = range(Δx, 2π, N_x)
    k = fftfreq(N_x, N_x)
    u = u_initial.(x)       # Impose initial conditions
    w_fft = fft(u)

    # Time step!
    u̲_return = [u]          # Set up return arrays (mostly for plotting after the fact)
    e(t) = exp.(-(sf^3).*im.*(k.^3).*t)
    f(t, w_fft, k) = - 0.5sf.*im.*k[1].*e(t).*fft(ifft(w_fft./e(t)).^2)
    @showprogress "Time stepping..." for t_i in t[1:end - 1]
        w_fft .+= RK_time_step_explicit(f, Δt, t_i, w_fft, [k]; RK_method = RK_forward_euler)
        u̲ = real(ifft(w_fft./e(t_i)))
        push!(u̲_return, u̲)
    end

    return (t, x, u̲_return)
end

""" Takes a solution on (0, 2π) and maps to a solution on (x_s, x_e). """
function KdV_solver_x_s_to_x_e(;
    u_initial::Function = u_1_INITIAL,
    t_e::Float64 = 1.0,
    x_s::Float64 = 0.0,
    x_e::Float64 = 2π,
    N_t::Int64 = 10_000,
    N_x::Int64 = 2^8
)
    sf = 2π/(x_e - x_s)     # Domain scale factor

    domain_map(y) = y/sf + x_s
    (t, y, u̲_return) = KdV_solver_0_to_2pi(
        u_initial = u_initial ∘ domain_map,
        t_e = t_e,
        sf = sf,
        N_t = N_t,
        N_x = N_x,
    )
    x = domain_map.(y)
    return (t, x, u̲_return)
end

function animate_KdV_equation_solution(;
    u_initial::Function = u_1_INITIAL,
    t_e::Float64 = 1.0,
    x_s::Float64 = 0.0,
    x_e::Float64 = 2π,
    N_t::Int64 = 10_000,
    N_x::Int64 = 2^8,
    max_frames::Int64 = 240
)
    (t, x, u_return) = KdV_solver_x_s_to_x_e(
        u_initial = u_initial,
        t_e = t_e,
        x_s = x_s,
        x_e = x_e,
        N_t = N_t,
        N_x = N_x
    )

    max_frames = (max_frames > N_t) ? N_t : max_frames  # Ensures at least 1 frame
    animation = @animate for (i, u) in enumerate(u_return)
        t_str = string(t[i])[1:min(length(string(t[i])), 4)]
        plot(
            x, u,
            title=latexstring("t = $t_str"),
            xlims=(x_s, x_e), ylims=(0, 12*0.3^2*1.2),
            size=(1200, 800)
        )
    end every fld(N_t, max_frames)

    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/animations/$dtstring-animation_KdV_RK4.mov"
    mov(animation, fname; fps=24)
end

function plot_KdV_equation_solution_in_time(;
    u_initial::Function = u_1_INITIAL,
    t_e::Float64 = 1.0,
    x_s::Float64 = 0.0,
    x_e::Float64 = 2π,
    N_t::Int64 = 10_000,
    N_x::Int64 = 2^8,
    max_series::Int64 = 10
)
    (t, x, u_return) = KdV_solver_x_s_to_x_e(
        u_initial = u_initial,
        t_e = t_e,
        x_s = x_s,
        x_e = x_e,
        N_t = N_t,
        N_x = N_x
    )
    number_series = (N_t < max_series) ? N_t : max_series

    time_series_plot = plot()
    for i in indices(number_series, N_t)
        plot!(time_series_plot, x, fill(t[i], size(x)), u_return[i])
    end

    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/animations/$dtstring-animation_KdV_RK4.mov"
    savefig(time_series_plot, fname)
end

function indices(samples, N)
    return 0
end

function plot_KdV_errors(;
    u_exact::Function = u_1_EXACT,
    u_initial::Function = u_1_INITIAL,
    t_e::Float64 = 1.0,
    x_s::Float64 = 0.0,
    x_e::Float64 = 2π,
    N_t_array::Vector{Int64} = collect(100:100:1_000),
    N_x::Int64 = 2^8
)
    Δt_array = []
    err_array = []
    for N_t in N_t_array
        # Run simulation for N_t time points
        (t, x, u_return) = KdV_solver_x_s_to_x_e(
            u_initial = u_initial,
            t_e = t_e,
            x_s = x_s,
            x_e = x_e,
            N_t = N_t,
            N_x = N_x
        )
        err = max(abs.(u_return[end] - u_exact.(t_e, x))...)   # Find max error over all x at t = t_e
        push!(Δt_array, t[2] - t[1])
        push!(err_array, err)
    end

    plot_err = plot(
        Δt_array, err_array,
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel="||E||",
        markershape=:circle, label=latexstring("Approximate solution with RK4"),
        key=:bottomright, size=(1600, 1200), thickness_scaling=2.0
    )
    for order in 1:4
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle=:dash, label=L"1\cdot (\Delta t)^%$order"
        )
    end
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/convergence/$dtstring-convergence_KdV_RK4.png"
    savefig(plot_err, fname)
end