using FFTW, Dates, ProgressMeter, LaTeXStrings, Plots

u_INITIAL_1(x) = exp(-100*(x - π)^2)
u_INITIAL_2(x) = max(0, 1 - abs(x/π - 1))
u_INITIAL_3(x) = exp(-100*(x - 1)^2)

"""
Solves the heat equation ∂ₜu = c∂ₓ²u by time stepping with Fourier spectral
methods.
Assumes spatial domain x ∈ (0, 2π] with periodic boundary conditions.
"""
function heat_equation_spectral_FE(;
    u_initial::Function=u_INITIAL_1,
    c=1,
    t_e=10.0,
    N_x=2^8,
    N_t=10_000
)
    t = range(0.0, t_e, N_t)
    Δt = t[2] - t[1]

    Δx = 2π/N_x
    x = range(x_s + Δx, x_e, N_x)

    k = fftfreq(N_x, N_x)
    k² = k.^2

    u = u_initial.(x)
    u_fft = fft(u)
    u_return = [u]

    @showprogress "Time stepping..." for _ in t[1:end - 1]
        u_fft .+= -c*Δt.*k².*u_fft      # Only the transformed values are needed every time step
        u = real(ifft(u_fft))
        push!(u_return, u)
    end

    return (t, x, u_return)
end

function animate_heat_equation_solution(; max_frames=240)
    (t, x, u_return) = heat_equation_spectral_FE(;
        u_initial=u_INITIAL_1,
        c=0.1,
        t_e=3.0,
        N_x=2^8,
        N_t=10_000
    )

    N_t = length(t)
    max_frames = (max_frames > N_t) ? N_t : max_frames # Ensures at least 1 frame
    animation = @animate for (i, uᵢ) in enumerate(u_return)
        t_str = string(t[i])[1:min(length(string(t[i])), 4)]
        plot(
            x, uᵢ,
            title=latexstring("t = $t_str"),
            xlims=(0.0, 2π), ylims=(0.0, 1.0),
            size=(1200, 800)
        )
    end every fld(N_t, max_frames)

    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/animations/$dtstring-animation_heat_equation_FE_goodfreq.mov"
    mov(animation, fname; fps=24)
end