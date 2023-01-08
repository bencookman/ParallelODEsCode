using Dates, Plots, BenchmarkTools, Statistics, LinearAlgebra, LaTeXStrings, Measures

include("ProjectTools.jl")

using .ProjectTools

"""
Find borders to stability regions (where Am(λΔt)≈1) at various corrections
levels.
"""
function corrections_stability_border(Δt, p, K; λ₀=Complex(-1.0, 0.0))
    θ_res = 100
    Δr = 0.01
    ∂S = zeros(Complex, θ_res, p)

    S = integration_matrix_equispaced(p - 1)

    r_max = 10
    θ = range(0, 2pi, θ_res) |> collect
    r = 0:Δr:r_max
    for (i, θᵢ) in enumerate(θ)
        λ_old = λ₀
        break_now = [false for _ in 1:p]
        for rⱼ in r
            λ = Complex(rⱼ*cos(θᵢ), rⱼ*sin(θᵢ)) + λ₀
            f(t, y) = λ*y
            (_, η) = RIDC_FE_sequential_correction_levels(S, f, 0, K*Δt, 1, K, K, p)
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

function plot_corrections_stability(Δt, p, K; λ₀=Complex(-1.0, 0.0))
    # Get border data
    ∂S = corrections_stability_border(Δt, p, K)
    ∂S = [val for val in ∂S]
    # Plot borders
    border_plot = plot(
        legend_position=(0.925, 0.225), aspect_ratio=:equal, size=(1200, 1000),
        xlims=(-2.5, 0.5), xlabel=latexstring("\$Re(\\lambda\\Delta t)\$"),
        ylims=(-1.5, 1.5), ylabel=latexstring("\$Im(\\lambda\\Delta t)\$"),
        left_margin=-25.0mm,
        thickness_scaling=3.0
    )
    colours = [:black, :red, :orange, :green, :blue, :purple]
    border_plot_color(l) = colours[(l - 1)%length(colours) + 1]
    border_plot_label(l) = (l == 1) ? "Prediction" : "Correction $(l)"
    for l in 1:p
        plot!(
            border_plot, real(∂S[:, l]), imag(∂S[:, l]),
            label=border_plot_label(l), color=border_plot_color(l)
        )
    end

    # Save and display plot
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/stability/stability-regions-RIDC_FE_$(p)_$(K)_$(N)-$dtstring"
    savefig(border_plot, fname*".png")
    savefig(border_plot, fname*".pdf")
    display(border_plot)
end

"""
Find the border to the stability region (where Am(λΔt)≈1) for an integration
scheme.
"""
function stability_border(Δt, p, K; λ₀=Complex(-1.0, 0.0))
    θ_res = 100
    Δr = 0.01
    ∂S = zeros(Complex, θ_res)

    S = integration_matrix_equispaced(p - 1)

    r_max = 10
    θ = range(0, 2pi, θ_res) |> collect
    r = 0:Δr:r_max
    for (i, θᵢ) in enumerate(θ)
        λ_old = λ₀
        for rⱼ in r
            λ = Complex(rⱼ*cos(θᵢ), rⱼ*sin(θᵢ)) + λ₀
            f(t, y) = λ*y
            (_, η) = RIDC_FE_sequential(S, f, 0, K*Δt, 1, K, K, p)

            amp = abs.(η[2])   # Amplification factor at all prediction/correction levels
            if amp > 1.0
                ∂S[i] = λ_old*Δt
                break
            end
            λ_old = λ
        end
    end

    return ∂S
end

function plot_stabilities(Δt, p, K; λ₀=Complex(-1.0, 0.0))
    # Get border data
    ∂S_array = [stability_border(Δt, p, Kᵢ) for Kᵢ in K]
    # ∂S = [val for val in ∂S]
    # Plot borders
    border_plot = plot(
        legend_position=(0.925, 0.225), aspect_ratio=:equal, size=(1200, 1000),
        xlims=(-2.5, 0.5), xlabel=latexstring("\$Re(\\lambda\\Delta t)\$"),
        ylims=(-1.5, 1.5), ylabel=latexstring("\$Im(\\lambda\\Delta t)\$"),
        left_margin=-25.0mm,
        thickness_scaling=3.0
    )
    colours = [:black, :red, :orange, :green, :blue, :purple]
    border_plot_color(i) = colours[(i - 1)%length(colours) + 1]
    for (i, Kᵢ) in enumerate(K)
        plot!(
            border_plot, real(∂S_array[i]), imag(∂S_array[i]),
            label="RIDC($(p), $(Kᵢ))", color=border_plot_color(i)
        )
    end

    # Save and display plot
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/stability/stability-regions-RIDC_FE_p=$(p)_K=$(min(K...))to$(max(K...))-$dtstring"
    savefig(border_plot, fname*".png")
    # savefig(border_plot, fname*".pdf")
    display(border_plot)
end

