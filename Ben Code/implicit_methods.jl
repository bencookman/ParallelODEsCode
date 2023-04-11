using
    Dates,
    Plots,
    BenchmarkTools,
    Statistics,
    LinearAlgebra,
    LaTeXStrings,
    FastGaussQuadrature,
    PyCall,
    ProgressMeter,
    Measures

include("ProjectTools.jl")

using .ProjectTools

const MATPLOTLIB = pyimport("matplotlib")
const RCPARAMS = PyDict(MATPLOTLIB["rcParams"])
RCPARAMS["mathtext.fontset"] = "cm"
RCPARAMS["xtick.major.pad"] = 10


## ROOT FINDING/FIXED POINT ITERATIONS

""" """
function Newton_iterate(
    vector_field,
    inverse_Jacobian_matrix,
    start_value;
    max_iterations = 10,
    tolerance = 1e-7
)
    current_value = copy(start_value)
    for _ in 1:max_iterations
        change_in_value = inverse_Jacobian_matrix(current_value)*vector_field(current_value)
        (sum(abs.(change_in_value)) < tolerance) && break
        current_value .-= change_in_value
    end
    return current_value
end

""" """
function Newton_iterate_1D(
    f,
    df,
    start_value::T;
    max_iterations = 10,
    epsilon = 1e-10,
    tolerance = 1e-7
) where {T <: Number}
    current_value = start_value
    for _ in 1:max_iterations
        df_value = df(current_value)
        (abs(df_value) < epsilon) && break
        change_in_value = f(current_value)/df(current_value)
        (abs(change_in_value) < tolerance) && break
        current_value -= change_in_value
    end
    return current_value
end

function backward_Euler_1D(
    ODE_system::ODESystem,
    ∂f∂y,
    N
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    for k in 1:N
        η[k + 1] = Newton_iterate_1D(
            (η_next -> η[k] + Δt*f(t[k + 1], η_next) - η_next),
            (η_next -> Δt*∂f∂y(t[k + 1], η_next) - 1),
            η[k];
            max_iterations = 10
        )
    end

    return (t, η)
end


## IMPLICIT METHODS

""" """
function IDC_Euler_implicit_1D(
    ODE_system::ODESystem,
    ∂f∂y,
    N, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    M = p - 1
    J = fld(N, M)
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            # Approximates η[k + 1] = η[k] + Δt*f(t[k + 1], η[k + 1])
            η[k + 1] = Newton_iterate_1D(
                (η_next -> η[k] + Δt*f(t[k + 1], η_next) - η_next),
                (η_next -> Δt*∂f∂y(t[k + 1], η_next) - 1),
                η[k]
            )
        end
        # Correction loop
        I = (j*M + 1):((j + 1)*M + 1)
        for _ in 2:p
            η_old = copy(η)
            for m in 1:M
                k = j*M + m
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
                # Approximates η[k + 1] = η[k] + Δt*(f(t[k + 1], η[k + 1]) - f(t[k + 1], η_old[k + 1])) + Δt*∫fₖ
                η[k + 1] = Newton_iterate_1D(
                    (η_next -> η[k] + Δt*(f(t[k + 1], η_next) - f(t[k + 1], η_old[k + 1])) + Δt*∫fₖ - η_next),
                    (η_next -> Δt*∂f∂y(t[k + 1], η_next) - 1),
                    η[k]
                )
            end
        end
    end

    return (t, η)
end
""" """
function IDC_Euler_implicit_1D_correction_levels(
    ODE_system::ODESystem,
    ∂f∂y,
    N, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    M = p - 1
    J = fld(N, M)
    η = zeros(typeof(y_s), N + 1, p)
    η[1, :] .= y_s

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            # Approximates η[k + 1] = η[k] + Δt*f(t[k + 1], η[k + 1])
            η[k + 1, 1] = Newton_iterate_1D(
                (η_next -> η[k, 1] + Δt*f(t[k + 1], η_next) - η_next),
                (η_next -> Δt*∂f∂y(t[k + 1], η_next) - 1),
                η[k, 1]
            )
        end
        # Correction loop
        I = (j*M + 1):((j + 1)*M + 1)
        for l in 2:p
            for m in 1:M
                k = j*M + m
                ∫fₖ = dot(S[m, :], f.(t[I], η[I, l - 1]))
                # Approximates η[k + 1] = η[k] + Δt*(f(t[k + 1], η[k + 1]) - f(t[k + 1], η_old[k + 1])) + Δt*∫fₖ
                η[k + 1, l] = Newton_iterate_1D(
                    (η_next -> η[k, l] + Δt*(f(t[k + 1], η_next) - f(t[k + 1], η[k + 1, l - 1])) + Δt*∫fₖ - η_next),
                    (η_next -> Δt*∂f∂y(t[k + 1], η_next) - 1),
                    η[k, l - 1]
                )
            end
        end
    end

    return (t, η)
end


## TESTS

function IDC_test_implicit_correction_levels()
    p = 4
    N_array = (1:3:100) .* (p - 1)
    S = integration_matrix_uniform(p - 1)

    orders_to_plot = 1:p

    ∂f∂y(t, y) = 4
    @unpack_ODETestSystem stiff_system_1
    @unpack_ODESystem ODE_system
    Δt_array = (t_e - t_s)./N_array
    levels_err_array = Array{Float64, 2}(undef, p, length(N_array))

    η_exact = y(t_e)
    for (i, N) in enumerate(N_array)
        (_, η_approx) = IDC_Euler_implicit_1D_correction_levels(
            ODE_system,
            ∂f∂y,
            N, p, S
        )
        η_out = real(η_approx[end, :])
        for l in axes(η_out)[1]
            err = err_rel(η_exact, η_out[l])
            levels_err_array[l, i] = err
        end
    end
    plot_err = plot(
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel="||E||",
        key=:bottomright, size=(1600, 1200), thickness_scaling=2.0
    )
    for l in axes(levels_err_array)[1]
        plot!(
            plot_err,
            Δt_array, levels_err_array[l, :],
            markershape=:circle, label=latexstring("Solution approximated with IDC-BE at level \$l = $l\$")
        )
    end
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle=:dash, label=L"1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

function IDC_test_implicit()
    p = 4
    N_array = (1:3:100) .* (p - 1)
    S = integration_matrix_uniform(p - 1)

    orders_to_plot = 1:p

    ∂f∂y(t, y) = 4
    @unpack_ODETestSystem stiff_system_1
    @unpack_ODESystem ODE_system
    Δt_array = (t_e - t_s)./N_array
    err_array = Vector{Float64}(undef, length(N_array))

    η_exact = y(t_e)
    for (i, N) in enumerate(N_array)
        (_, η_approx) = backward_Euler_1D(
            ODE_system,
            ∂f∂y,
            N
        )
        η_out = real(η_approx[end, end])
        err = err_rel(η_exact, η_out)
        err_array[i] = err
    end
    plot_err = plot(
        xscale = :log10, yscale = :log10, xlabel = L"Δt", ylabel = "||E||",
        key = :bottomright, size = (1600, 1200), thickness_scaling = 2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape = :circle, label = "Solution approximated with BE",
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle = :dash, label = L"1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)
end

