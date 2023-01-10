using FFTW, Dates, ProgressMeter, LaTeXStrings, Plots

u_INITIAL_1(x) = sech(10*(x - 1))^2

function KDV_equation_spectral_FE(;
    u_initial::Function=u_INITIAL_1,
    c=6,
    t_e=0.1,
    x_s=0.0,
    x_e=2π,
    N_x=2^9,
    N_t=1_000
)
    t = range(0.0, t_e, N_t)
    Δt = t[2] - t[1]
    Δx = 2π/N_x
    x = range(Δx, 2π, N_x)
    k = fftfreq(N_x, N_x)
    u = u_initial.(x)       # Impose initial conditions
    û = fft(u)

    # Time step!
    u_return = [u]  # Set up return arrays (mostly for plotting after the fact)
    f(k, û; c=c) = (
        im.*(k.^3).*û
        .-
        c.*fft(ifft(û).*ifft(im.*k.*û))
    )
    @showprogress "Time stepping..." for _ in t[1:end - 1]
        # Forward Euler time step
        k₁ = f(k, û)
        k₂ = f(k, û .+ Δt.*k₁.*0.5)
        k₃ = f(k, û .+ Δt.*k₂.*0.5)
        k₄ = f(k, û .+ Δt.*k₃)
        û .+= Δt.*(k₁ .+ 2k₂ .+ 2k₃ .+ k₄)

        u = real(ifft(û))
        push!(u_return, u)
    end

    return (t, x, u_return)
end

function animate_KDV_solution(; max_frames=240)
    (t, x, u_return) = KDV_equation_spectral_FE(u_initial=u_INITIAL_1)

    N_t = length(t)
    max_frames = (max_frames > N_t) ? N_t : max_frames  # Ensures at least 1 frame
    animation = @animate for (i, u) in enumerate(u_return)
        t_str = string(t[i])[1:min(length(string(t[i])), 4)]
        plot(
            x, u,
            title=latexstring("t = $t_str"),
            xlims=(0, 2π),
            size=(1200, 800)
        )
    end every fld(N_t, max_frames)

    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/animations/animation_KDV_FE-$dtstring.mov"
    mov(animation, fname; fps=24)
end