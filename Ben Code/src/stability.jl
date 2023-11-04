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
    scheme, scale;
    θ_res = 100, Δr = 0.01
)
    integrator = scheme[1]
    parameters = scheme[2]
    λ₀ = scheme[3]

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
                1.0/scale,
                1.0 + 0.0im
            )
            (_, η) = integrator(ODE_system, parameters...)


            is_z_stable = (abs(η[end][end]) < 1.0)
            if !is_z_stable
                ∂S[i] = λ_old
                break
            end
            λ_old = λ
        end
    end

    return ∂S
end

function plot_schemes_polar_stability_borders(
    schemes_scales_names;
    θ_res = 100, Δr = 0.01,
    x_range = (-2.5, 0.5), y_range = (-2.0, 2.0),
    x_ticks = [-2.0, -1.0, 0.0], y_ticks = [-2.0, -1.0, 0.0, 1.0, 2.0]
)
    # Get border data
    ∂S_array = [
        find_schemes_polar_stability_border(
            scheme, scale;
            θ_res = θ_res, Δr = Δr
        )
        for (scheme, scale, _) in schemes_scales_names
    ]
    # Plot borders
    border_plot = plot(
        aspect_ratio = :equal,
        xlims = x_range, ylims = y_range,
        xticks = x_ticks, yticks = y_ticks,
        xlabel = "Re(z)",
        ylabel = "Im(z)",
        thickness_scaling = 4.0,
        legend = (1, 0),
    )
    colours = [:black, :red, :orange, :green, :blue, :purple]
    border_plot_color(i) = colours[(i - 1)%length(colours) + 1]
    for (i, ∂S) in enumerate(∂S_array)
        plot!(
            border_plot, real(∂S), imag(∂S),
            color = border_plot_color(i), label = schemes_scales_names[i][3],
        )
    end

    # Save and display plot
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/stability/$dtstring-stability_border_polar"
    # save_formats = [".png", ".pdf"]
    # for save_format in save_formats
    #     savefig(border_plot, fname*save_format)
    # end
    display(border_plot)
end

"""
Uses a Cartesian discretisation of the complex plane and returns two matrices:
λ_mesh         = those values λ on the complex plane which where analysed,
stability_mesh = whether or not the corresponding value on λ was stable or not.
"""
function find_schemes_cartesian_stability_mesh(
    scheme, scale;
    x_values = x_values, y_values = y_values
)
    integrator = scheme[1]
    parameters = scheme[2]

    # Create 2D meshes
    λ_mesh = [x + im*y for y in y_values, x in x_values]
    stability_mesh = fill(true, size(λ_mesh))

    # Calculate stability mesh
    @showprogress "Running Cartesian stability analysis on scheme..." for i in 1:length(y_values), j in 1:length(x_values)
        # Run approximation for this λ
        λ = λ_mesh[i, j]
        ODE_system = ODESystem(
            (t, η) -> λ*η,
            1.0/scale,
            1.0 + 0.0im
        )
        (t, η) = integrator(ODE_system, parameters...)

        # Is this λ value stable or not
        η_end = η[end][end]
        is_z_stable = (abs(η_end) < 1.0)
        stability_mesh[i, j] = is_z_stable
    end

    return stability_mesh
end

"""
Store scheme parameters like
[(scheme_1_name, (scheme_1_parameters, ...)), ...] ~ Vector{Tuple{scheme, named tuple of parameters}}
"""
function plot_schemes_cartesian_stability_meshes(
    schemes_scales;
    x_res = 100, y_res = 100,
    x_range = (-2.5, 0.5), y_range = (-2.0, 2.0),
    x_ticks = [-2.0, -1.0, 0.0], y_ticks = [-2.0, -1.0, 0.0, 1.0, 2.0]
)
    # Find stability regions for each of the schemes
    x_values = range(x_range[1], x_range[2], x_res)
    y_values = range(y_range[1], y_range[2], y_res)
    stability_meshes = []
    for (scheme, scale) in schemes_scales
        stability_mesh = find_schemes_cartesian_stability_mesh(
            scheme, scale;
            x_values = x_values, y_values = y_values
        )
        push!(stability_meshes, stability_mesh)
    end

    # Plot these stability regions
    stability_to_value(stability; stable_value = 1.0, unstable_value = NaN) = (
        stability ? stable_value : unstable_value
    )
    stability_heatmap = heatmap(
        aspect_ratio = :equal, colorbar = false,
        thickness_scaling = 4.0,
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
            fillalpha = 0.35, seriescolor = cgrad(:rainbow)
        )
    end
    display(stability_heatmap)
end

function stability_test_RK()
    schemes_scales_names = [
        (
            (
                RK1_forward_euler,
                (1,),
                -1.0 + 0.0im
            ),
            1.0,
            "FE"
        ),
        (
            (
                RK2_Heuns,
                (1,),
                -1.0 + 0.0im
            ),
            1.0,
            "RK2"
        ),
        (
            (
                RK3_Kutta,
                (1,),
                -1.0 + 0.0im
            ),
            1.0,
            "RK3"
        ),
        (
            (
                RK4_standard,
                (1,),
                -1.0 + 0.0im
            ),
            1.0,
            "RK4"
        ),
    ]
    θ_res = 500     # Angular resolution
    Δr = 0.002      # Radial step
    x_range = (-3.5, 1.5)
    y_range = (-3.5, 3.5)
    x_ticks = [-3, -2, -1, 0, 1]
    y_ticks = [-3, -2, -1, 0, 1, 2, 3]
    plot_schemes_polar_stability_borders(
        schemes_scales_names;
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

function stability_test_IDC_corrections()
    schemes_scales = [
        (
            (
                RK1_forward_euler,
                (1,)
            ),
            1
        ),
        (
            (
                IDC_FE,
                (1, INTEGRATION_MATRIX_ARRAY_UNIFORM[2], 1)
            ),
            1.0
        ),
        (
            (
                IDC_FE,
                (2, INTEGRATION_MATRIX_ARRAY_UNIFORM[3], 1)
            ),
            1
        ),
        (
            (
                IDC_FE,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 1)
            ),
            1
        ),
        (
            (
                IDC_FE,
                (4, INTEGRATION_MATRIX_ARRAY_UNIFORM[5], 1)
            ),
            1.0
        ),
    ]
    x_res = 800
    y_res = 800
    x_range = (-6.5, 2)
    y_range = (-6, 6)
    x_ticks = [-9, -6, -3, 0]
    y_ticks = [-9, -6, -3, 0, 3, 6, 9]
    plot_schemes_cartesian_stability_meshes(
        schemes_scales;
        x_res = x_res, y_res = y_res,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

function stability_test_4th_order()
    schemes_scales = [
        (
            (
                RK4_standard,
                (1,)
            ),
            1.0
        ),
        (
            (
                IDC_FE,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 1)
            ),
            1.0
        ),
        (
            (
                IDC_RK2_Heuns,
                (1, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 1)
            ),
            1.0
        ),
    ]
    x_res = 500
    y_res = 500
    x_range = (-8, 2)
    y_range = (-6, 6)
    x_ticks = [-8, -4, 0]
    y_ticks = [-6, -3, 0, 3, 6]
    plot_schemes_cartesian_stability_meshes(
        schemes_scales;
        x_res = x_res, y_res = y_res,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

function stability_test_8th_order()
    schemes_scales = [
        (
            (
                RK8_Cooper_Verner,
                (1,)
            ),
            1.0
        ),
        (
            (
                IDC_FE,
                (7, INTEGRATION_MATRIX_ARRAY_UNIFORM[8], 1)
            ),
            1.0
        ),
        (
            (
                IDC_RK2_Heuns,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[8], 1)
            ),
            1.0
        ),
        (
            (
                IDC_RK4,
                (1, INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS[8], 1, interpolation_polynomials((0:7)./7))
            ),
            1.0
        ),
    ]
    x_res = 100
    y_res = 100
    x_range = (-25, 1)
    y_range = (-25, 25)
    x_ticks = [-20, -10, 0]
    y_ticks = [-20, -10, 0, 10, 20]
    plot_schemes_cartesian_stability_meshes(
        schemes_scales;
        x_res = x_res, y_res = y_res,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

function stability_test_groups()
    schemes_with_names = [
        (
            (
                IDC_FE,
                (1, INTEGRATION_MATRIX_ARRAY_UNIFORM[2], 1),
                1,
                -0.5 + 0.0im
            ),
            "1 group"
        ),
        (
            (
                IDC_FE,
                (1, INTEGRATION_MATRIX_ARRAY_UNIFORM[2], 20),
                20,
                -0.5 + 0.0im
            ),
            "20 groups"
        ),
    ]
    θ_res = 500     # Angular resolution
    Δr = 0.002      # Radial step
    x_range = (-2.5, 0.5)
    y_range = (-2, 2)
    x_ticks = [-3, -2, -1, 0, 1]
    y_ticks = [-3, -2, -1, 0, 1, 2, 3]
    plot_schemes_polar_stability_borders(
        schemes_with_names;
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

function stability_test_IDC_SDC_lobatto()
    schemes_scales_names = [
        (
            (
                IDC_FE,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 1),
                -2.0 + 0.0im
            ),
            1.0,
            "IDC4-FE"
        ),
        (
            (
                IDC_FE,
                (9, INTEGRATION_MATRIX_ARRAY_UNIFORM[10], 1),
                -5.0 + 0.0im
            ),
            1.0,
            "IDC10-FE"
        ),
        (
            (
                SDC_FE_lobatto,
                (3, INTEGRATION_MATRIX_ARRAY_LOBATTO[4], 1),
                -1.0 + 0.0im
            ),
            1.0,
            "SDC4-FE"
        ),
        (
            (
                SDC_FE_lobatto,
                (9, INTEGRATION_MATRIX_ARRAY_LOBATTO[10], 1),
                -1.0 + 0.0im
            ),
            1.0,
            "SDC10-FE"
        ),
    ]
    # x_res = 100
    # y_res = 100
    θ_res = 200
    Δr = 0.005
    x_range = (-12, 2)
    y_range = (-9, 9)
    x_ticks = [-12, -8, -4, 0]
    y_ticks = [-8, -4, 0, 4, 8]
    plot_schemes_polar_stability_borders(
        schemes_scales_names;
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
    # plot_schemes_cartesian_stability_meshes(
    #     schemes_scales_names;
    #     x_res = x_res, y_res = y_res,
    #     x_range = x_range, y_range = y_range,
    #     x_ticks = x_ticks, y_ticks = y_ticks
    # )
end

function stability_test_IDC_SDC_legendre()
    schemes_scales_names = [
        (
            (
                IDC_FE,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 1),
                -2.0 + 0.0im
            ),
            1.0,
            "IDC4-FE"
        ),
        (
            (
                IDC_FE,
                (9, INTEGRATION_MATRIX_ARRAY_UNIFORM[10], 1),
                -5.0 + 0.0im
            ),
            1.0,
            "IDC10-FE"
        ),
        (
            (
                SDC_FE_legendre,
                (3, INTEGRATION_MATRIX_ARRAY_LEGENDRE[4], 1),
                -2.0 + 0.0im
            ),
            1.0,
            "SDC4-FE"
        ),
        (
            (
                SDC_FE_lobatto,
                (9, INTEGRATION_MATRIX_ARRAY_LEGENDRE[10], 1),
                -5.0 + 0.0im
            ),
            1.0,
            "SDC10-FE"
        ),
    ]
    # x_res = 100
    # y_res = 100
    θ_res = 100
    Δr = 0.02
    x_range = (-12, 2)
    y_range = (-9, 9)
    x_ticks = [-12, -8, -4, 0]
    y_ticks = [-8, -4, 0, 4, 8]
    plot_schemes_polar_stability_borders(
        schemes_scales_names;
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
    # plot_schemes_cartesian_stability_meshes(
    #     schemes_scales_names;
    #     x_res = x_res, y_res = y_res,
    #     x_range = x_range, y_range = y_range,
    #     x_ticks = x_ticks, y_ticks = y_ticks
    # )
end

function stability_test_PCIDC4_groups()
    schemes_scales_names = reverse([
        (
            (
                PCIDC_FE,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 1),
            ),
            1.0
        ),
        (
            (
                PCIDC_FE,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 2),
            ),
            1.0/2
        ),
        (
            (
                PCIDC_FE,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 10),
            ),
            1.0/10
        ),
        (
            (
                PCIDC_FE,
                (3, INTEGRATION_MATRIX_ARRAY_UNIFORM[4], 100),
            ),
            1.0/100
        ),
    ])
    x_res = 300
    y_res = 300
    # θ_res = 100
    # Δr = 0.02
    x_range = (-7, 1)
    y_range = (-5, 5)
    x_ticks = [-6, -3, 0]
    y_ticks = [-4, -2, 0, 2, 4]
    # plot_schemes_polar_stability_borders(
    #     schemes_scales_names;
    #     θ_res = θ_res, Δr = Δr,
    #     x_range = x_range, y_range = y_range,
    #     x_ticks = x_ticks, y_ticks = y_ticks
    # )
    plot_schemes_cartesian_stability_meshes(
        schemes_scales_names;
        x_res = x_res, y_res = y_res,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

function stability_test_PCSDC4_groups()
    schemes_scales_names = reverse([
        (
            (
                PCSDC_FE_lobatto_reduced_quadrature,
                (3, INTEGRATION_MATRIX_ARRAY_LOBATTO[3], 1),
            ),
            1.0
        ),
        (
            (
                PCSDC_FE_lobatto_reduced_quadrature,
                (3, INTEGRATION_MATRIX_ARRAY_LOBATTO[3], 2),
            ),
            1.0/2
        ),
        (
            (
                PCSDC_FE_lobatto_reduced_quadrature,
                (3, INTEGRATION_MATRIX_ARRAY_LOBATTO[3], 10),
            ),
            1.0/10
        ),
        (
            (
                PCSDC_FE_lobatto_reduced_quadrature,
                (3, INTEGRATION_MATRIX_ARRAY_LOBATTO[3], 100),
            ),
            1.0/100
        ),
    ])
    x_res = 250
    y_res = 250
    # θ_res = 100
    # Δr = 0.02
    x_range = (-4.5, 1)
    y_range = (-3.5, 3.5)
    x_ticks = [-4, -2, 0]
    y_ticks = [-2, 0, 2]
    # plot_schemes_polar_stability_borders(
    #     schemes_scales_names;
    #     θ_res = θ_res, Δr = Δr,
    #     x_range = x_range, y_range = y_range,
    #     x_ticks = x_ticks, y_ticks = y_ticks
    # )
    plot_schemes_cartesian_stability_meshes(
        schemes_scales_names;
        x_res = x_res, y_res = y_res,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
end

function stability_test_PCIDC4_reduced_stencil_L()
    L_array = [1, 5, 20, 50]
    number_corrections = 3
    J = 1
    params_tuple = [get_PCIDC_FE_reduced_stencil_args(number_corrections, L) for L in L_array]

    schemes_scales_names = [
        (
            (
                PCIDC_FE_reduced_stencil,
                (number_corrections, params_tuple[i][1], J, L, params_tuple[i][2]),
                -2.0 + 0.0im
            ),
            1.0/J,
            string("L = ", L)
        )
        for (i, L) in enumerate(L_array)
    ]
    # x_res = 500
    # y_res = 500
    θ_res = 400
    Δr = 0.005
    x_range = (-10, 1)
    y_range = (-9, 9)
    x_ticks = [-8, -4, 0]
    y_ticks = [-8, -4, 0, 4, 8]
    plot_schemes_polar_stability_borders(
        schemes_scales_names;
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
    # plot_schemes_cartesian_stability_meshes(
    #     schemes_scales_names;
    #     x_res = x_res, y_res = y_res,
    #     x_range = x_range, y_range = y_range,
    #     x_ticks = x_ticks, y_ticks = y_ticks
    # )
end

function stability_test_PCSDC4_reduced_stencil_increased_quadrature_L()
    L_array = [1, 5, 20, 50]
    number_corrections = 3
    J = 1
    params_tuple = [get_PCIDC_FE_reduced_stencil_args(number_corrections, L) for L in L_array]

    schemes_scales_names = [
        (
            (
                PCSDC_FE_lobatto_reduced_stencil,
                (number_corrections, params_tuple[i][1], J, L, params_tuple[i][2]),
                -2.5 + 0.0im
            ),
            1.0/J,
            string("L = ", L)
        )
        for (i, L) in enumerate(L_array)
    ]
    # x_res = 500
    # y_res = 500
    θ_res = 400
    Δr = 0.005
    x_range = (-10, 1)
    y_range = (-9, 9)
    x_ticks = [-8, -4, 0]
    y_ticks = [-8, -4, 0, 4, 8]
    plot_schemes_polar_stability_borders(
        schemes_scales_names;
        θ_res = θ_res, Δr = Δr,
        x_range = x_range, y_range = y_range,
        x_ticks = x_ticks, y_ticks = y_ticks
    )
    # plot_schemes_cartesian_stability_meshes(
    #     schemes_scales_names;
    #     x_res = x_res, y_res = y_res,
    #     x_range = x_range, y_range = y_range,
    #     x_ticks = x_ticks, y_ticks = y_ticks
    # )
end

