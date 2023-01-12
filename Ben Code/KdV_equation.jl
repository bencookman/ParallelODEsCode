using FFTW, Dates, ProgressMeter, LaTeXStrings, Plots

include("ProjectTools.jl")

using .ProjectTools

u_INITIAL_1(x) = 12*0.3^2*(sech(0.3(x + 20)))^2

""" Solve the KdV equation using spectral methods on the region (0, 2π). """
function KdV_solver_0_to_2pi(;
    u_initial::Function = u_INITIAL_1,
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
        w_fft .+= RK_time_step_explicit(f, Δt, t_i, w_fft, [k]; RK_method = RK4_standard)
        u̲ = real(ifft(w_fft./e(t_i)))
        push!(u̲_return, u̲)
    end

    return (t, x, u̲_return)
end

function KdV_solver_x_s_to_x_e(;
    u_initial::Function = u_INITIAL_1,
    t_e::Float64 = 1.0,
    x_s::Float64 = 0.0,
    x_e::Float64 = 2π,
    N_t::Int64 = 10_000,
    N_x::Int64 = 2^8
)
    sf = 2π/(x_e - x_s)
    (t, y, u̲_return) = KdV_solver_0_to_2pi(
        u_initial = ((x) -> u_initial((x - x_s)*sf)),
        t_e = t_e,
        sf = sf,
        N_t = N_t,
        N_x = N_x,
    )
    x = y./sf .+ x_s
    return (t, x, u̲_return)
end

function animate_KdV_equation_solution(;
    u_initial::Function = u_INITIAL_1,
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