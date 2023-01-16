using FFTW, Dates, ProgressMeter, LaTeXStrings, Plots

include("ProjectTools.jl")

using .ProjectTools

u_SOLITON_EXACT(t, x; k = 1.0, x_0 = 0.0) = 12*k^2*(sech(k*(x - x_0) - 4*k^3*t))^2

u_1_EXACT(t, x) = u_SOLITON_EXACT(t, x; k = 0.3, x_0 = -20.0)   # On (-15π, 15π)
u_1_INITIAL(x) = u_1_EXACT(0, x)
u_2_EXACT(t, x) = u_SOLITON_EXACT(t, x; k = 0.2, x_0 = -5.0)     # On (-15π, 15π)
u_2_INITIAL(x) = u_2_EXACT(0, x)

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
    u_return = [u]          # Set up return arrays (mostly for plotting after the fact)
    # e(t) = exp.(-(sf^3).*im.*(k.^3).*t)
    # f(t, w_fft, k) = - 0.5sf.*im.*k[1].*e(t).*fft(ifft(w_fft./e(t)).^2)
    # f(t, w_fft, k) = - 0.5sf.*im.*k.*e(t).*fft(ifft(w_fft./e(t)).^2)
    f(w_fft, k, e) = - 0.5sf.*im.*k.*e.*fft(ifft(w_fft./e).^2)
    @showprogress "Time stepping..." for t_i in t[1:end - 1]
        # k₁ = f(t_i, w_fft, k)
        # k₂ = f(t_i + 0.5Δt, w_fft + 0.5Δt*k₁, k)
        # k₃ = f(t_i + 0.5Δt, w_fft + 0.5Δt*k₂, k)
        # k₄ = f(t_i + Δt, w_fft + Δt*k₃, k)
        e = exp.(-(sf^3).*im.*(k.^3).*t_i)
        k₁ = f(w_fft, k, e)
        k₂ = f(w_fft + 0.5Δt*k₁, k, e)
        k₃ = f(w_fft + 0.5Δt*k₂, k, e)
        k₄ = f(w_fft + Δt*k₃, k, e)
        w_fft .+= (k₁ + 2k₂ + 2k₃ + k₄)*Δt/6

        # w_fft .+= RK_time_step_explicit(f, Δt, t_i, w_fft, [k]; RK_method = RK_forward_euler)
        # u̲ = real(ifft(w_fft./e(t_i)))
        u = real(ifft(w_fft./e))
        push!(u_return, u)
    end

    return (t, x, u_return)
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
    max_height = 1.0,
    max_series::Int64 = 20
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

    # time_series_plot = plot()
    # for i in get_sample_indices(N_t, number_series)
    #     plot!(
    #         time_series_plot,
    #         fill(t[i], size(x)), x, u_return[i],
    #         linecolor = :black, series_type
    #         legend = false, camera = (80, 40),
    #         xflip = true, zlims = (0.0, max_height),
    #         fillrange = 0.0
    #     )
    # end
    sample_indices = get_sample_indices(N_t, number_series)
    t_sampled = t[sample_indices]
    t_mesh = [t_value for t_value in t_sampled, x_value in x]
    x_mesh = [x_value for t_value in t_sampled, x_value in x]
    z_surface = Surface([u_return[i][j] for i in sample_indices, j in 1:length(x)])
    time_series_plot = surface(
        t_mesh,
        x_mesh,
        z_surface,
        # series_type = :surface,# seriescolor = :black,
        # surface_colour = [:white for __ in x, _ in t_sampled], fill_colour = :match,
        legend = false, camera = (80, 40),
        xflip = true, zlims = (0.0, max_height),
        fillrange = 0.0
    )

    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/$dtstring-time_series_KdV_RK4"
    # savefig(time_series_plot, "$fname.png")
    # savefig(time_series_plot, "$fname.pdf")
end

"""
Out of N available series of data, return the indices of <samples> series'
including 1 and N which are as evenly spaced as possible.
"""
function get_sample_indices(N, samples)
    indices_float = (N - 1)./(samples - 1).*(0.0:(samples - 1)) .+ 1
    return round.(Int64, indices_float)
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