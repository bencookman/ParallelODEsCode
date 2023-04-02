using
    Dates,
    Plots,
    BenchmarkTools,
    Statistics,
    LinearAlgebra,
    LaTeXStrings,
    Measures,
    ProgressMeter,
    PyCall

include("ProjectTools.jl")

using .ProjectTools

const MATPLOTLIB = pyimport("matplotlib")
const RCPARAMS = PyDict(MATPLOTLIB["rcParams"])
RCPARAMS["mathtext.fontset"] = "cm"
RCPARAMS["xtick.major.pad"] = 10

"""
Find borders to stability regions (where Am(λΔt)≈1) at various corrections
levels.
"""
function find_schemes_correction_levels_polar_stability_borders(
    scheme,
    λ₀,
    correction_levels_indices;
    Δt = 1.0,
    θ_res = 100, Δr = 0.01,
    r_max = 10
)
    ∂S = zeros(Complex, θ_res, size(correction_levels_indices)[1])
    θ = range(0, 2pi, θ_res) |> collect
    r = 0:Δr:r_max
    @showprogress "Running polar stability analysis on scheme for all correction levels..." for (i, θᵢ) in enumerate(θ)
        λ_old = λ₀
        break_now = [false for l in axes(correction_levels_indices)[1]]
        for rⱼ in r
            λ = Complex(rⱼ*cos(θᵢ), rⱼ*sin(θᵢ)) + λ₀
            ODE_system = ODESystem(
                (t, y) -> λ*y,
                Δt*scheme[2].N,
                1.0 + 0.0im
            )
            (_, η) = scheme[1](ODE_system, scheme[2]...)

            amp = abs.(η[end - 1, :])   # Amplification factor at all prediction/correction levels
            for l in axes(correction_levels_indices)[1]
                # Do not adjust values if that level has already been dealt with
                if !break_now[l] && amp[correction_levels_indices[l]] > 1.0
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

function plot_schemes_correction_levels_polar_stability_borders(
    scheme_λ₀,
    correction_levels_indices;
    Δt = 1.0,
    θ_res = 100, Δr = 0.01,
    x_range = (-2.5, 0.5), y_range = (-2.0, 2.0),
    x_ticks = [-2.0, -1.0, 0.0], y_ticks = [-2.0, -1.0, 0.0, 1.0, 2.0]
)
    # Get border data
    ∂S = find_schemes_correction_levels_polar_stability_borders(
        scheme_λ₀[1],
        scheme_λ₀[2],
        correction_levels_indices;
        Δt = Δt,
        θ_res = θ_res, Δr = Δr
    )
    # ∂S = [val for val in ∂S]
    # Plot borders
    border_plot = plot(
        legend_position = (1.15, 0.4), aspect_ratio = :equal, size = (1200, 1000),
        xlims = x_range, ylims = y_range,
        xticks = x_ticks, yticks = y_ticks,
        xlabel = latexstring("\$Re(\\lambda\\Delta t)\$"),
        ylabel = latexstring("\$Im(\\lambda\\Delta t)\$"),
        left_margin = -30.0mm, right_margin = 20.0mm,
        thickness_scaling = 3.0
    )
    colours = [:black, :red, :orange, :green, :blue, :purple]
    border_plot_color(l) = colours[(l - 1)%length(colours) + 1]
    border_plot_label(l) = (l == 1) ? "Prediction" : "Correction $(l)"
    for l in axes(correction_levels_indices)[1]
        plot!(
            border_plot, real(∂S[:, l]), imag(∂S[:, l]),
            label=border_plot_label(correction_levels_indices[l]), color=border_plot_color(l)
        )
    end

    # Save and display plot
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/stability/$dtstring-stability-border-polar"
    # savefig(border_plot, fname*".png")
    # savefig(border_plot, fname*".pdf")
    display(border_plot)
end

function run_correction_levels_polar_stability_borders()
    # S = integration_matrix_equispaced(3)
    # scheme_λ₀ = ((
    #     RIDC_FE_sequential_correction_levels,
    #     NamedTuple{(:N, :K, :p, :S)}((3, 3, 4, S))
    # ),  -0.5 + 0.0im
    # )
    N = 3
    p = 2
    S = integration_matrix_legendre(p)
    scheme_λ₀ = ((
    #     SDC_FE_correction_levels,
    #     NamedTuple{(:N, :p, :S)}((N, p, S))
    # ),  -0.75 + 0.0im,
        SDC_FE_correction_levels,
        NamedTuple{(:N, :p, :S)}((N, p, S))
    ),  -1.0 + 0.0im,
    )
    correction_levels_indices = 1:p#[1, 2, 3, 4, 10]
    Δt = 1.0
    θ_res = 500
    Δr = 0.002
    x_range = (-3.0, 0.5)
    y_range = (-2.0, 2.0)
    x_ticks = [-2.0, 0.0]
    y_ticks = [-2.0, 0.0, 2.0]
    plot_schemes_correction_levels_polar_stability_borders(
        scheme_λ₀,
        correction_levels_indices;
        Δt = Δt,
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end


"""
Find the border to the stability region (where Am(λΔt)≈1) for an integration
scheme.
"""
function find_schemes_polar_stability_border(
    scheme;
    Δt = 1.0,
    θ_res = 100, Δr = 0.01
)
    integrator = scheme[1]
    parameters = scheme[2]
    N =  scheme[3]
    λ₀ = scheme[4]

    ∂S = zeros(Complex, θ_res)
    r_max = 10
    θ = range(0, 2pi, θ_res)
    r = 0:Δr:r_max
    @showprogress "Running polar stability analysis on scheme..." for (i, θᵢ) in enumerate(θ)
        λ_old = λ₀
        for rⱼ in r
            λ = Complex(rⱼ*cos(θᵢ), rⱼ*sin(θᵢ)) + λ₀

            ODE_system = ODESystem(
                (t, y) -> λ*y,
                Δt*N,
                1.0 + 0.0im
            )
            (_, η) = integrator(ODE_system, parameters...)

            is_λ_stable = (abs.(η[end]) < 1.0)
            if !is_λ_stable
                ∂S[i] = λ_old*Δt
                break
            end
            λ_old = λ
        end
    end

    return ∂S
end

function plot_schemes_polar_stability_borders(
    schemes_with_names;
    Δt = 1.0,
    θ_res = 100, Δr = 0.01,
    x_range = (-2.5, 0.5), y_range = (-2.0, 2.0),
    x_ticks = [-2.0, -1.0, 0.0], y_ticks = [-2.0, -1.0, 0.0, 1.0, 2.0]
)
    # Get border data
    ∂S_array = [
        find_schemes_polar_stability_border(
            scheme;
            Δt = Δt,
            θ_res = θ_res, Δr = Δr
        )
        for (scheme, _) in schemes_with_names
    ]
    # Plot borders
    border_plot = plot(
        aspect_ratio = :equal, size = (900, 900),
        xlims = x_range, ylims = y_range,
        xticks = x_ticks, yticks = y_ticks,
        xlabel = "Re(z)",
        ylabel = "Im(z)",
        # left_margin = -25.0mm,
        thickness_scaling = 4.0,
        legend = (0.7, 0.05),
        guidefontsize = 10, tickfontsize = 8, legendfontsize = 8
    )
    colours = [:black, :red, :orange, :green, :blue, :purple]
    border_plot_color(i) = colours[(i - 1)%length(colours) + 1]
    for (i, ∂S) in enumerate(∂S_array)
        plot!(
            border_plot, real(∂S), imag(∂S),
            color = border_plot_color(i), label = schemes_with_names[i][2]
        )
    end

    # Save and display plot
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/stability/$dtstring-stability_border_polar"
    save_formats = [".png", ".pdf"]
    for save_format in save_formats
        savefig(border_plot, fname*save_format)
    end
    display(border_plot)
end

function run_polar_stability_borders()
    lobatto_schemes = [
        (
            (
                SDC_FE_lobatto,
                NamedTuple{(:number_corrections, :S, :J)}(
                    (number_corrections, integration_matrix_lobatto(ceil(Int64, number_corrections/2) + 2), 1)
                ),
                ceil(Int64, number_corrections/2) + 1,
                -0.5 + 0.0im
            ),
            "p = $(number_corrections + 1)"
        )
        for number_corrections in 1:3:10
    ]
    schemes_with_names = [
        (
            (
                RK1_forward_euler,
                (1,),
                1,
                -1.0 + 0.0im
            ),
            ""
        ),
        lobatto_schemes...
    ]
    Δt = 1.0
    θ_res = 500     # Angular resolution
    Δr = 0.002      # Radial step
    x_range = (-3.5, 1.5)
    y_range = (-3.5, 3.5)
    x_ticks = [-3, -2, -1, 0, 1]
    y_ticks = [-3, -2, -1, 0, 1, 2, 3]
    plot_schemes_polar_stability_borders(
        schemes_with_names;
        Δt = Δt,
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

"""
Uses a Cartesian discretisation of the complex plane and returns two matrices:
λ_mesh         = those values λ on the complex plane which where analysed,
stability_mesh = whether or not the corresponding value on λ was stable or not.
"""
function find_schemes_cartesian_stability_mesh(
    scheme;
    Δt = 1.0,
    x_values = x_values, y_values = y_values
)
    # Create 2D meshes
    λ_mesh = [x + im*y for y in y_values, x in x_values]
    stability_mesh = fill(true, size(λ_mesh))

    # Calculate stability mesh
    @showprogress "Running Cartesian stability analysis on scheme..." for i in 1:length(y_values), j in 1:length(x_values)
        # Run approximation for this λ
        λ = λ_mesh[i, j]
        ODE_system = ODESystem(
            (t, η) -> λ*η,
            Δt*scheme[3],
            1.0 + 0.0im
        )
        (_, η) = scheme[1](ODE_system, scheme[2]...)
        # Is this λ value stable or not
        is_λ_stable = (abs(η[end]) < 1.0)
        stability_mesh[i, j] = is_λ_stable
    end

    return (λ_mesh, stability_mesh)
end

"""
Store scheme parameters like
[(scheme_1_name, (scheme_1_parameters, ...)), ...] ~ Vector{Tuple{scheme, named tuple of parameters}}
"""
function plot_schemes_cartesian_stability_meshes(
    schemes;
    Δt = 1.0,
    x_res = 100, y_res = 100,
    x_range = (-2.5, 0.5), y_range = (-2.0, 2.0),
    x_ticks = [-2.0, -1.0, 0.0], y_ticks = [-2.0, -1.0, 0.0, 1.0, 2.0]
)
    # Find stability regions for each of the schemes
    x_values = range(x_range[1], x_range[2], x_res)
    y_values = range(y_range[1], y_range[2], y_res)
    stability_meshes = []
    for scheme in schemes
        (_, stability_mesh) = find_schemes_cartesian_stability_mesh(
            scheme;
            Δt = Δt,
            x_values = x_values, y_values = y_values
        )
        push!(stability_meshes, stability_mesh)
    end

    # Plot these stability regions
    stability_to_value(stability; stable_value = 1.0, unstable_value = NaN) = (
        stability ? stable_value : unstable_value
    )
    stability_heatmap = heatmap(
        aspect_ratio = :equal, colorbar = false, legend = true,
        thickness_scaling = 3.0, xguide = "x",
        xlims = x_range, ylims = y_range,
        xticks = x_ticks, yticks = y_ticks,
        xlabel="Re(z)",
        ylabel="Im(z)",
    )
    for (i, stability_mesh) in enumerate(stability_meshes)
        heatmap!(
            stability_heatmap,
            x_values, y_values,
            stability_to_value.(stability_mesh; stable_value = i, unstable_value = NaN),
            fillalpha = 0.5, seriescolor = cgrad(:lighttest),
        )
    end
    display(stability_heatmap)
end

function run_cartesian_stability_meshes()
    # number_corrections = 11
    # M = number_corrections
    # M = 4*(number_corrections + 1) + 1
    # S = integration_matrix_uniform(M)
    # S = integration_matrix_legendre(M - 1)
    # S = integration_matrix_legendre_RK4(M - 1)
    lobatto_schemes = [
        (
            SDC_FE_lobatto,
            NamedTuple{(:number_corrections, :S, :J)}(
                (number_corrections, integration_matrix_lobatto(ceil(Int64, number_corrections/2) + 2), 1)
            ),
            ceil(Int64, number_corrections/2) + 1
        )
        for number_corrections in 1:5
    ]
    schemes = [
        (
            RK1_forward_euler,
            NamedTuple{(:N,)}((1,)),
            1
        ),
        lobatto_schemes...
    ]
    Δt = 1.0
    x_res = 100
    y_res = 100
    x_range = (-4, 2)
    y_range = (-4, 4)
    # x_ticks = [0.0, 10.0, 20.0, 30.0, 40.0]
    # y_ticks = [-20.0, -10.0, 0.0, 10.0, 20.0]
    plot_schemes_cartesian_stability_meshes(
        schemes;
        Δt = Δt,
        x_res = x_res, y_res = y_res,
        x_range = x_range, y_range = y_range
        # x_ticks = x_ticks, y_ticks = y_ticks
    )
end



function polar_stability_borders_SDC_FE_lobatto()
    lobatto_schemes = [
        (
            (
                SDC_FE_lobatto,
                NamedTuple{(:number_corrections, :S, :J)}(
                    (number_corrections, integration_matrix_lobatto(ceil(Int64, number_corrections/2) + 2), 1)
                ),
                ceil(Int64, number_corrections/2) + 1,
                -0.5 + 0.0im
            ),
            "p = $(number_corrections + 1)"
        )
        for number_corrections in 1:3:10
    ]
    schemes_with_names = [
        (
            (
                RK1_forward_euler,
                (1,),
                1,
                -1.0 + 0.0im
            ),
            ""
        ),
        lobatto_schemes...
    ]
    Δt = 1.0
    θ_res = 500     # Angular resolution
    Δr = 0.002      # Radial step
    x_range = (-3.5, 1.5)
    y_range = (-3.5, 3.5)
    x_ticks = [-3, -2, -1, 0, 1]
    y_ticks = [-3, -2, -1, 0, 1, 2, 3]
    plot_schemes_polar_stability_borders(
        schemes_with_names;
        Δt = Δt,
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

function polar_stability_borders_SDC_FE_legendre()
    legendre_schemes = [
        (
            (
                SDC_FE_legendre,
                NamedTuple{(:number_corrections, :S, :J)}(
                    (number_corrections, integration_matrix_legendre(number_corrections + 2), 1)
                ),
                number_corrections + 1,
                -0.5 + 0.0im
            ),
            "p = $(number_corrections + 1)"
        )
        for number_corrections in 1:3:10
    ]
    schemes_with_names = [
        (
            (
                RK1_forward_euler,
                (1,),
                1,
                -1.0 + 0.0im
            ),
            "FE"
        ),
        legendre_schemes...
    ]
    Δt = 1.0
    θ_res = 500     # Angular resolution
    Δr = 0.002      # Radial step
    x_range = (-3.5, 1.5)
    y_range = (-3.5, 3.5)
    x_ticks = [-3, -2, -1, 0, 1]
    y_ticks = [-3, -2, -1, 0, 1, 2, 3]
    plot_schemes_polar_stability_borders(
        schemes_with_names;
        Δt = Δt,
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end
