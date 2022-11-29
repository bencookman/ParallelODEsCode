using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra, LaTeXStrings

include("ProjectTools.jl")

using .ProjectTools

function corrections_stability_border(Δt, p, K, N)
    θ_res = 1000
    Δr = 0.001
    ∂S = zeros(Complex, θ_res, p)

    S = integration_matrix_equispaced(p - 1)

    r_max = 10
    θ = range(0, 2pi, θ_res) |> collect
    r = 0:Δr:r_max
    for (i, θᵢ) in enumerate(θ)
        λ_old = 0.0 + 0.0im
        break_now = [false for _ in 1:p]
        for rⱼ in r
            λ = Complex(rⱼ*cos(θᵢ), rⱼ*sin(θᵢ))
            f(t, y) = λ*y
            η = RIDC_FE_sequential_matrix(S, f, 0, N*Δt, 1, N, K, p)
            amp = abs.(η[2, :])   # Amplification factor at all prediction/correction levels

            for l in 1:p
                # Do not adjust values if that level has already been dealt with
                if !break_now[l] && amp[l] > 1.0
                    ∂S[i, l] = λ_old*Δt
                    break_now[l] = true
                end
            end
            all(break_now) && break     # Only go to next θ when all levels are ready
            λ_old = λ
        end
    end

    return ∂S
end

function plot_corrections_stability(Δt, p, K, N)
    ∂S = corrections_stability_border(Δt, p, K, N)
    ∂S = [val for val in ∂S]
    border_plot = plot(∂S[:, 1], label="Prediction", color=:black)
    plot!(border_plot, ∂S[:, 2], label="1st Correction", color=:red)
    plot!(border_plot, ∂S[:, 3], label="2nd Correction", color=:orange)
    plot!(border_plot, ∂S[:, 4], label="3rd Correction", color=:green)
    plot!(
        legend_position=(0.925, 0.225), aspect_ratio=:equal, size=(1800, 1200),
        xlims=(-2.5, 0.5), xlabel=latexstring("\$Re(\\lambda\\Delta t)\$"),
        ylims=(-1.5, 1.5), ylabel=latexstring("\$Im(\\lambda\\Delta t)\$"),
        thickness_scaling=3.0
    )
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/stability-regions-RIDC_$(p)_$(K)_$(N)-$dtstring"
    savefig(border_plot, fname*".png")
    savefig(border_plot, fname*".pdf")
    display(border_plot)
end