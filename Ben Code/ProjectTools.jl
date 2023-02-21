"""
Key:
IDC -> Integral Deferred Correction
RIDC -> Revisionist Integral Deferred Correction
ODE -> Ordinary Differential equation
FE -> Forward Euler time-stepping scheme
RK -> Runge-Kutta multi-stage time-stepping scheme
"""
module ProjectTools

using
    LinearAlgebra,
    Statistics,
    PyCall,
    FastGaussQuadrature,
    Parameters,
    BarycentricInterpolation

export
    err_abs,
    err_rel,
    err_cum,
    err_norm,
    err_max,

    ODESystem,
    @unpack_ODESystem,
    ODETestSystem,
    @unpack_ODETestSystem,
    Butcher_p53_system,
    sqrt_system,
    cube_system,
    stiff_system_1,

    RKMethod,
    @unpack_RKMethod,
    RK_time_step_explicit,
    RK4_standard,
    RK2_midpoint,
    RK2_trapezoid,
    RK1_forward_euler,
    RK8_Cooper_Verner,

    integration_matrix,
    integration_matrix_uniform,
    integration_matrix_legendre,

    IDC_FE,
    IDC_FE_correction_levels,
    IDC_RK2_trapezoid,
    IDC_RK_general,
    get_IDC_RK_group_poly,
    IDC_FE_single,
    IDC_FE_single_correction_levels,
    SDC_FE,
    SDC_FE_correction_levels,
    SDC_FE_single,
    RIDC_FE_sequential,
    RIDC_FE_sequential_correction_levels,
    RIDC_FE_sequential_reduced_stencil,
    RIDC_RK2_trapeoid_sequential,

    Newton_iterate,
    Newton_iterate_1D,
    backward_Euler_1D,
    IDC_Euler_implicit_1D,
    IDC_Euler_implicit_1D_correction_levels


err_abs(exact, approx) = abs.(exact - approx)
err_rel(exact, approx) = err_abs(exact, approx) ./ abs.(exact)
err_cum(exact, approx) = [sum(err_abs(exact[1:i], approx[1:i])) for i in 1:length(data_correct)]
err_norm(exact, approx, p) = norm(err_abs(exact, approx), p)
err_max(exact, approx)  = maximum(err_abs(exact, approx))


"""
Structure to contain an ODE system with no exact solution
    f::Function  - gradient function, y' = f(t, y)
    t_s::Float64 - start time, y(t_s) is the solution at the initial time
    t_e::Float64 - end time, y(t_e) is the solution at the final time
    y_s::Float64 - starting value y(t_s) = y_s
"""
@with_kw struct ODESystem{T<:Function, S<:Number}
    f::T
    t_s::Float64
    t_e::Float64
    y_s::S
    ODESystem{T, S}(f, t_s, t_e, y_s) where {T<:Function, S<:Number} = (t_s < t_e) ? new(f, t_s, t_e, y_s) : error("time t_s must be less than time t_e")
end

# Point(x::T, y::T) where {T<:Real} = Point{T}(x,y)
ODESystem(f::T, t_s, t_e, y_s::S) where {T<:Function, S<:Number} = ODESystem{T, S}(f, t_s, t_e, y_s)
ODESystem(f, t_e, y_s) = ODESystem(f, 0.0, t_e, y_s)

"""
Structure to contain a test ODE system which has an exact solution
    ODE_system::ODESystem - The base ODE system to solve
    y::Function - exact solution
"""
@with_kw struct ODETestSystem{T<:Function}
    ODE_system::ODESystem
    y::T
end

ODETestSystem(f, t_s, t_e, y_s, y) = ODETestSystem(ODESystem(f, t_s, t_e, y_s), y)
ODETestSystem(f, t_e, y_s, y) = ODETestSystem(ODESystem(f, 0.0, t_e, y_s), y)

""" Taken from page 53 of Numerical Methods for ODEs by J C Butcher """
const Butcher_p53_system = ODETestSystem(
    (t, y) -> (y - 2t*y^2)/(1 + t),
    1.0,
    0.4,
    t -> (1 + t)/(t^2 + 1/0.4)
)
""" https://doi.org/10.1137/09075740X """
const sqrt_system = ODETestSystem(
    (t, y) -> 4t*sqrt(y),
    5.0,
    1.0 + 0.0im,
    t -> (1 + t^2)^2
)
const cube_system = ODETestSystem(
    (t, y) -> t^3,
    5.0,
    2.0,
    t -> 0.25*t^4 + 2.0,
)

const stiff_system_1 = ODETestSystem(
    (t, y) -> 4y,
    3.0,
    1.0,
    t -> exp(4t)
)


"""
Represents an arbitrary Runge-Kutta method using its Butcher Tableau. Contains:
    a::Array{T, 2} - coefficients of summations of kⱼ terms in calculation of kᵣ
    b::Array{T, 1} - coefficients of summations of kⱼ terms in calculation of y_{i+1}
    c::Array{T, 1} - coefficients of time step in calculation of kᵣ
Note: First dimension of a must match length of c and second dimension of a must
match length of b.
"""
@with_kw struct RKMethod
    a::Array{<:Real, 2}
    b::Array{<:Real, 1}
    c::Array{<:Real, 1}
    order_of_accuracy::Int64
    function RKMethod(a, b, c, order_of_accuracy)
        check_RK_method_size(a, b, c)
        (length(axes(c)[1]) >= order_of_accuracy)  || error("System too small to support this order of accuracy")
        return new(a, b, c, order_of_accuracy)
    end
end

function check_RK_method_size(a, b, c)
    (length(axes(a)[1]) == length(axes(c)[1])) || error("a and c axes do not match")
    (length(axes(a)[2]) == length(axes(b)[1])) || error("a and b axes do not match")
end

"""
Perform a single explicit Runge-Kutta time step using scheme given:

u_{i+1} = uᵢ + Δt⋅∑bᵣ⋅kᵣ, where
kᵣ = f(tᵢ + cᵣ⋅Δt, uᵢ + Δt⋅∑aᵣₗ⋅kₗ, other_coords)

( k = [k₁, k₂, ..., kₛ] )

Note how extra coordinates in the system, e.g. x and y, are to be bundled together in
vector as argument other_coords.
"""
function RK_time_step_explicit(
    f::Function,
    Δt::Float64,
    t_i::Float64,
    u_i,
    other_coords...;
    RK_method::RKMethod = RK4_standard
)
    @unpack_RKMethod RK_method

    k = [f(t_i, u_i, other_coords...)]
    for r in axes(b)[1][2:end]
        k_sum = zeros(typeof(u_i[1]), size(u_i))
        for l in axes(b)[1][1]:(r - 1)
            k_sum .+= a[r, l].*k[l]
        end
        push!(k, f(t_i + c[r]*Δt, u_i .+ Δt.*k_sum, other_coords...))
    end
    k_sum_full = zeros(typeof(u_i[1]), size(u_i))
    for r in axes(b)[1]
        k_sum_full .+= b[r].*k[r]
    end
    return Δt.*k_sum_full
end
"""
Perform a single explicit correction time-step step using the given Runge-Kutta scheme:

η^{l+1}_{i+1} = η^{l+1}ᵢ + Δt⋅∑bᵣ⋅kᵣ + Δt⋅Σ, where
kᵣ = f(tᵢ + cᵣ⋅Δt, η^{l+1}ᵢ + Δt⋅∑aᵣₗ⋅kₗ + cᵣ⋅Δt⋅Σ) - f(tᵢ + cᵣ⋅Δt, ηˡ(tᵢ + cᵣ⋅Δt))

( k = [k₁, k₂, ..., kₛ] )

η_old is the interpolant function η^{l+1}(t) evaluated at each time tᵢ + cᵣ⋅Δt.
"""
function RK_correction_time_step_explicit(
    f::Function,
    Δt::Float64,
    t_i::Float64,
    η_i,
    Σ,
    η_old;
    RK_method::RKMethod = RK4_standard
)
    @unpack_RKMethod RK_method

    k = [f(t_i, η_i) - f(t_i, η_old[1])]
    for r in axes(b)[1][2:end]
        k_sum = zeros(typeof(η_i[1]), size(η_i))
        for l in axes(b)[1][1]:(r - 1)
            k_sum .+= a[r, l].*k[l]
        end
        k_next = f(t_i + c[r]*Δt, η_i .+ Δt.*k_sum .+ c[r].*Σ) - f(t_i + c[r]*Δt, η_old[r])
        push!(k, k_next)
    end
    k_sum_full = zeros(typeof(η_i[1]), size(η_i))
    for r in axes(b)[1]
        k_sum_full .+= b[r].*k[r]
    end
    return Δt.*k_sum_full .+ Δt.*Σ
end

""" Canonical RK4 method. """
RK4_standard = RKMethod(
    a = [
        0//1  0//1  0//1  0//1;
        1//2  0//1  0//1  0//1;
        0//1  1//2  0//1  0//1;
        0//1  0//1  1//1  0//1
    ], b = [
        1//6, 1//3, 1//3, 1//6
    ], c = [
        0//1, 1//2, 1//2, 1//1
    ],
    order_of_accuracy = 4
)
""" Midpoint Runge-Kutta method. """
RK2_midpoint = RKMethod(
    a = [
        0//1  0//1;
        1//2  0//1
    ], b = [
        0//1, 1//1
    ], c = [
        0//1, 1//2
    ],
    order_of_accuracy = 2
)
""" Second order trapezoidal Runge-Kutta method (A.K.A. Heun's method) """
RK2_trapezoid = RKMethod(
    a = [
        0//1  0//1;
        1//1  0//1
    ], b = [
        1//2, 1//2
    ], c = [
        0//1, 1//1
    ],
    order_of_accuracy = 2
)
""" Forward Euler method. """
RK1_forward_euler = RKMethod(
    a = fill(0, 1, 1),
    b = [1],
    c = [0],
    order_of_accuracy = 1
)
RK8_Cooper_Verner = RKMethod(
    a = Float64[
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.25 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        1/7 (-7 - 3sqrt(21))/98 (21 + 5sqrt(21))/49 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        (11 + sqrt(21))/84 0.0 (18 + 4sqrt(21))/63 (21 - sqrt(21))/252 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        (5 + sqrt(21))/48 0.0 (9 + sqrt(21))/36 (-231 + 14sqrt(21))/360 (63 - 7sqrt(21))/80 0.0 0.0 0.0 0.0 0.0 0.0;
        (10 - sqrt(21))/42 0.0 (-432 + 92sqrt(21))/315 (633 - 145sqrt(21))/90 (-504 + 115sqrt(21))/70 (63 - 13sqrt(21))/35 0.0 0.0 0.0 0.0 0.0;
        1/14 0.0 0.0 0.0 0.0 (14 - 3sqrt(21))/126 (13 - 3sqrt(21))/63 1/9 0.0 0.0 0.0;
        1/32 0.0 0.0 0.0 0.0 (91 - 21sqrt(21))/576 11/72 (-385 - 75sqrt(21))/1152 (63 + 13sqrt(21))/128 0.0 0.0;
        1/14 0.0 0.0 0.0 0.0 1/9 (-733 - 147sqrt(21))/2205 (515 + 111sqrt(21))/504 (-51 - 11sqrt(21))/56 (132 + 28sqrt(21))/245 0.0;
        0.0 0.0 0.0 0.0 0.0 (-42 + 7sqrt(21))/18 (-18 + 28sqrt(21))/45 (-273 - 53sqrt(21))/72 (301 + 53sqrt(21))/72 (28 - 28sqrt(21))/45 (49 - 7sqrt(21))/18
    ], b = Float64[
        1/20, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 49/180, 16/45,
        49/180, 1/20
    ], c = Float64[
        0.0, 0.5, 0.5,
        (7 + sqrt(14))/14, (7 + sqrt(14))/14, 0.5,
        (7 - sqrt(21))/14, (7 - sqrt(21))/14, 0.5,
        (7 + sqrt(21))/14, 1.0
    ],
    order_of_accuracy = 8
)


function RK_general(
    ODE_system::ODESystem,
    N;
    RK_method::RKMethod
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    for k in 1:N
        η[k + 1] = η[k] + RK_time_step_explicit(
            f, Δt, t[k], η[k];
            RK_method = RK_method
        )
    end

    return (t, η)
end

""" Any RKMethod is callable to solve a given ODE_system over N time intervals  """
(RK_method::RKMethod)(ODE_system::ODESystem, N) = RK_general(ODE_system, N; RK_method = RK_method)


"""  """
function integration_matrix(t_quadrature_nodes, t_time_steps)
    scipy_interpolate = pyimport("scipy.interpolate")
    # Calculate anti-derivatives of polynomial interpolants
    q_func = []
    for i in axes(t_quadrature_nodes)[1]
        yᵢ = zeros(Float64, size(t_quadrature_nodes))
        yᵢ[i] = 1.0
        pᵢ_coeffs = scipy_interpolate.lagrange(t_quadrature_nodes, yᵢ).c |> reverse # Lagrange interpolant to yᵢ at t
        qᵢ_coeffs = pᵢ_coeffs ./ (1.0:length(pᵢ_coeffs))                            # Antiderivative of pᵢ
        qᵢ_func(x) = sum(qᵢⱼ*x^j for (j, qᵢⱼ) in enumerate(qᵢ_coeffs))
        push!(q_func, qᵢ_func)
    end
    # Use antiderivatives to evaluate integration matrix
    return [
        qᵢ_func(t_time_steps[i + 1]) - qᵢ_func(t_time_steps[i])
        for i in axes(t_time_steps)[1][1:end - 1], qᵢ_func in q_func
    ]
end

"""  """
function integration_matrix_uniform(N)
    t_nodes = collect(0.0:N)
    return integration_matrix(t_nodes, t_nodes)
end

"""  """
function integration_matrix_legendre(N)
    gl_quad = gausslegendre(N)
    return integration_matrix(gl_quad[1], vcat(-1, gl_quad[1], 1))
end


"""
Algorithm from https://doi.org/10.1137/09075740X
Note passing integration matrix S as argument is much more efficient for testing
as the same matrix may otherwise be calculated multiple times for different
function calls.

An integration matrix S = integration_matrix_equispaced(p - 1) should be used
"""
function IDC_FE(
    ODE_system::ODESystem,
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
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        I = (j*M + 1):((j + 1)*M + 1)
        for _ in 2:p
            η_old = copy(η)
            for m in 1:M
                k = j*M + m
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
        end
    end

    return (t, η)
end
function IDC_FE(
    ODE_system::ODESystem,
    J, M, number_corrections, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    N = J*M
    p = number_corrections + 1
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        I = (j*M + 1):((j + 1)*M + 1)
        for _ in 1:number_corrections
            η_old = copy(η)
            for m in 1:M
                k = j*M + m
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
        end
    end

    return (t, η)
end

"""
Same algorithm as 'IDC_FE' but storing all prediction and correction
levels in a matrix output. This is useful for calculating stability regions at
different correction stages.

An integration matrix S = integration_matrix_equispaced(p - 1) should be used
"""
function IDC_FE_correction_levels(
    ODE_system::ODESystem,
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
            η[k + 1, 1] = η[k, 1] + Δt*f(t[k], η[k, 1])
        end
        # Correction loop
        for l in 2:p
            for m in 1:M
                k = j*M + m
                I = (j*M + 1):((j + 1)*M + 1)
                ∫fₖ = dot(S[m, :], f.(t[I], η[I, l - 1]))
                η[k + 1, l] = η[k, l] + Δt*(f(t[k], η[k, l]) - f(t[k], η[k, l - 1])) + Δt*∫fₖ
            end
        end
    end

    return (t, η)
end

"""
An integration matrix S = integration_matrix_equispaced(p - 1) should be used
"""
function IDC_RK2_trapezoid(
    ODE_system::ODESystem,
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
            η[k + 1] = RK_time_step_explicit(f, Δt, t[k], η[k]; RK_method = RK2_trapezoid)
        end
        # Correction loop
        I = (j*M + 1):((j + 1)*M + 1)   # Quadrature nodes
        for _ in 2:fld(p, 2)
            η_old = copy(η)
            for m in 1:M
                k = j*M + m
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
                η[k + 1] = RK_correction_time_step_explicit(
                    f, Δt, t[k], η[k], ∫fₖ, [η_old[k], η_old[k + 1]];
                    RK_method = RK2_trapezoid
                )
            end
        end
    end

    return (t, η)
end

"""
An integration matrix S = integration_matrix_equispaced(p - 1) should be used
"""
function IDC_RK_general(
    ODE_system::ODESystem,
    N, p, S;
    RK_method::RKMethod = RK2_midpoint
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
            η[k + 1] = RK_time_step_explicit(f, Δt, t[k], η[k]; RK_method = RK_method)
        end
        # Correction loop
        I = (j*M + 1):((j + 1)*M + 1)   # Quadrature nodes
        for _ in 2:fld(p, 2)
            η_old = copy(η)
            η_old_group_poly = get_IDC_RK_group_poly(η_old, t[j*M + 1], t[(j + 1)*M], (j*M + 1):((j + 1)*M))
            for m in 1:M
                k = j*M + m
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
                η_old_interpolants = [η_old_group_poly(t[k] + Δt*cᵣ) for cᵣ in RK_method.c]
                η[k + 1] = RK_correction_time_step_explicit(
                    f, Δt, t[k], η[k], ∫fₖ, η_old_interpolants;
                    RK_method = RK_method
                )
            end
        end
    end

    return (t, η)
end

"""
Generates the polynomial which interpolates η_old at all points in a given group
and returns as a function of time.
"""
function get_IDC_RK_group_poly(η_old, t_group_s, t_group_e, group_indices)
    equispaced_poly = Equispaced{length(group_indices) - 1}()
    interpolating_poly = interpolate(equispaced_poly, η_old[group_indices])
    return (t_in -> interpolating_poly(2(t_in - t_group_s)/(t_group_e - t_group_s) - 1))
end

"""
Integral deferred correction with a single subinterval (uses static 
interpolatory quadrature over all N + 1 nodes). To improve this could use a
reduced size integration matrix of minimum necessary size (p - 1 = M) with
rolling usage of interpolation nodes.

An integration matrix S = integration_matrix_equispaced(N) is used
"""
function IDC_FE_single(
    ODE_system::ODESystem,
    N, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    # Prediction loop
    for m in 1:N
        η[m + 1] = η[m] + Δt*f(t[m], η[m])
    end
    # Correction loop
    I = 1:(N + 1)
    for _ in 2:p
        η_old = copy(η)
        for m in 1:N
            ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
            η[m + 1] = η[m] + Δt*(f(t[m], η[m]) - f(t[m], η_old[m])) + Δt*∫fₖ
        end
    end

    return (t, η)
end

"""
Same algorithm as 'IDC_FE_single' but storing all prediction and correction
levels. This is useful for calculating stability regions at different correction
stages.

An integration matrix S = integration_matrix_equispaced(N) is used
"""
function IDC_FE_single_correction_levels(
    ODE_system::ODESystem,
    N, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    η = zeros(typeof(y_s), N + 1, p)
    η[1, :] .= y_s

    # Prediction loop
    for m in 1:N
        η[m + 1, 1] = η[m, 1] + Δt*f(t[m], η[m, 1])
    end
    # Correction loop
    for l in 2:p
        for m in 1:N
            ∫fₖ = dot(S[m, :], f.(t[:], η[:, l - 1]))
            η[m + 1, l] = η[m, l] + Δt*(f(t[m], η[m, l]) - f(t[m], η[m, l - 1])) + Δt*∫fₖ
        end
    end

    return (t, η)
end

"""
To use the more stable legendre nodes, we must integrate over different regions
of our subintervals compared to where the interpolatory legendre nodes are. This
changes the integration matrix and subsequent indexing.

In order for our quadrature to have sufficient order by the last correction
step, we need to use the same number of nodes as the order we wish to acheive
with this step.

It is not possible to make a reduced stencil algorithm for this as our legendre
nodes are fixed within their subintervals.

An integration matrix S = integration_matrix_legendre(p) should be used
"""
function SDC_FE(
    ODE_system::ODESystem,
    N, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = p + 1
    J = fld(N, M)
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s
    (t, group_scale) = calculate_legendre_time_discretisation(t_s, t_e, p, J)
    Δt = t[2:end] .- t[1:end - 1]   # No longer constant

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1] = η[k] + Δt[k]*f(t[k], η[k])
        end
        # Correction loop
        I = (j*M + 2):((j + 1)*M)   # Use non-endpoint nodes for quadrature
        for _ in 2:p
            η_old = copy(η)
            for m in 1:M
                k = j*M + m
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))*group_scale/2
                η[k + 1] = η[k] + Δt[k]*(f(t[k], η[k]) - f(t[k], η_old[k])) + ∫fₖ
            end
        end
    end

    return (t, η)
end

""" Sandwich legendre nodes within subintervals """
function calculate_legendre_time_discretisation(t_s, t_e, number_of_legendre_nodes, J)
    legendre_quadrature = gausslegendre(number_of_legendre_nodes)
    group_scale = (t_e - t_s)/J
    legendre_t = (legendre_quadrature[1] .+ 1).*group_scale/2
    t = []
    for j in 0:(J - 1)
        push!(t, j*group_scale)
        push!(t, (j*group_scale .+ legendre_t)...)
    end
    push!(t, t_e)
    return (t, group_scale)
end

function SDC_FE_correction_levels(
    ODE_system::ODESystem,
    N, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = p + 1
    J = fld(N, M)   # Require N | (p + 1)
    η = zeros(typeof(y_s), N + 1, p)
    η[1, :] .= y_s
    (t, group_scale) = calculate_legendre_time_discretisation(t_s, t_e, p, J)
    Δt = t[2:end] .- t[1:end - 1]    # No longer constant

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1, 1] = η[k, 1] + Δt[k, 1]*f(t[k], η[k, 1])
        end
        # Correction loop
        I = (j*M + 2):((j + 1)*M)   # Use non-endpoint nodes for quadrature
        for l in 2:p
            for m in 1:M
                k = j*M + m
                ∫fₖ = dot(S[m, :], f.(t[I], η[I, l - 1]))*group_scale/2
                η[k + 1, l] = η[k, l] + Δt[k]*(f(t[k], η[k, l]) - f(t[k], η[k, l - 1])) + ∫fₖ
            end
        end
    end

    return (t, η)
end

"""
An integration matrix S = integration_matrix_legendre(N - 1) should be used
"""
function SDC_FE_single(
    ODE_system::ODESystem,
    N, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s
    (t, _) = calculate_legendre_time_discretisation(t_s, t_e, N - 1, 1)
    gl = gausslegendre(N - 1)
    t_legendre = (gl[1] .+ 1)/2                     # legendre nodes in [0, 1]
    t_legendre = t_legendre.*(t_e - t_s) .+ t_s     # Legendre nodes in [a, b]
    t = vcat(t_s, t_legendre, t_e)
    Δt = t[2:end] .- t[1:end - 1]                   # No longer constant

    # Prediction loop
    for k in 1:N
        η[k + 1] = η[k] + Δt[k]*f(t[k], η[k])
    end
    # Correction loop
    I = 2:N                                         # Use non-endpoint nodes for quadrature
    for _ in 2:p
        η_old = copy(η)
        for k in 1:N
            ∫fₖ = dot(S[k, :], f.(t[I], η_old[I]))*(t_e - t_s)/2
            η[k + 1] = η[k] + Δt[k]*(f(t[k], η[k]) - f(t[k], η_old[k])) + ∫fₖ
        end
    end

    return (t, η)
end

"""
The sequential form of our RIDC algorithm is relatively simple: we've simply
added more space in each subinterval whilst interpolating over the same number
of nodes. The main drawback to this potential parallelisability is that only the
less stable uniform nodes may be used.

An integration matrix S = integration_matrix_equispaced(p - 1) should be used
"""
function RIDC_FE_sequential(
    ODE_system::ODESystem,
    N, K, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    M = p - 1
    J = fld(N, K)
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    for j in 0:(J-1)
        # Prediction loop
        for m in 1:K
            k = j*K + m
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        for _ in 2:p
            η_old = copy(η)
            for m in 1:M
                k = j*K + m
                I = (j*K + 1):(j*K + M + 1)
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
            for m in (M + 1):K
                k = j*K + m
                I = (k + 1 - M):(k + 1)
                ∫fₖ = dot(S[M, :], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
        end
    end

    return (t, η)
end

"""
An integration matrix S = integration_matrix_equispaced(p - 1) should be used
"""
function RIDC_FE_sequential_correction_levels(
    ODE_system::ODESystem,
    N, K, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    M = p - 1
    J = fld(N, K)
    η = zeros(typeof(y_s), N + 1, p)
    η[1, :] .= y_s

    for j in 0:(J-1)
        # Prediction loop
        for m in 1:K
            k = j*K + m
            η[k + 1, 1] = η[k, 1] + Δt*f(t[k], η[k, 1])
        end
        # Correction loop
        for l in 2:p
            for m in 1:M
                k = j*K + m
                I = (j*K + 1):(j*K + M + 1)
                ∫fₖ = dot(S[m, :], f.(t[I], η[I, l - 1]))
                η[k + 1, l] = η[k, l] + Δt*(f(t[k], η[k, l]) - f(t[k], η[k, l - 1])) + Δt*∫fₖ
            end
            for m in (M + 1):K
                k = j*K + m
                I = (k + 1 - M):(k + 1)
                ∫fₖ = dot(S[M, :], f.(t[I], η[I, l - 1]))
                η[k + 1, l] = η[k, l] + Δt*(f(t[k], η[k, l]) - f(t[k], η[k, l - 1])) + Δt*∫fₖ
            end
        end
    end

    return (t, η)
end

"""
For forward Euler, at each correction step we are only 'aiming' for an order
equal to the number of corrections previously done (plus 2). In lieu of this, we
only require interpolatory quadrature involving a number of nodes again related
to # of previous correction steps.

An integration matrix S = [integration_matrix_equispaced(l) for l in 1:(p - 1)]
should be used
"""
function RIDC_FE_sequential_reduced_stencil(
    ODE_system::ODESystem,
    N, K, p, S::Array{Matrix{Float64}}
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    M = p - 1
    J = fld(N, K)
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    for j in 0:(J-1)
        # Prediction loop
        for m in 1:M
            k = j*K + m
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        for l in 1:M
            η_old = copy(η)
            for m in 1:M
                k = j*K + m
                I = (j*K + 1):(j*K + M + 1)
                ∫fₖ = dot(S[l][m, :], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
            for m in (M + 1):K
                k = j*K + m
                I = (j*K + m - M):(j*K + m)
                ∫fₖ = dot(S[l][M, :], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
        end
    end

    return (t, η)
end

"""  """
function RIDC_RK2_trapeoid_sequential(
    ODE_system::ODESystem,
    N, K, p, S
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    M = p - 1
    J = fld(N, K)
    η = zeros(typeof(y_s), N + 1)
    η[1] = y_s

    for j in 0:(J-1)
        # Prediction loop
        for m in 1:K
            k = j*K + m
            K₁ = f(t[k], η[k])
            K₂ = f(t[k + 1], η[k] .+ Δt*K₁)
            η[k + 1] = η[k] .+ 0.5Δt.*(K₁ .+ K₂)
        end
        # Correction loop
        for _ in 2:fld(p, 2)
            η_old = copy(η)
            for m in 1:M
                k = j*K + m
                I = (j*K + 1):(j*K + M + 1)
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
                K₁ = f(t[k], η[k]) .- f(t[k], η_old[k])
                K₂ = f(t[k + 1], η[k] .+ Δt.*(K₁ .+ ∫fₖ)) .- f(t[k + 1], η_old[k + 1])
                η[k + 1] = η[k] .+ 0.5Δt.*(K₁ .+ K₂) .+ Δt.*∫fₖ
            end
            for m in (M + 1):K
                k = j*K + m
                I = (j*K + m - M + 1):(j*K + m + 1)
                ∫fₖ = dot(S[M, :], f.(t[I], η_old[I]))
                K₁ = f(t[k], η[k]) .- f(t[k], η_old[k])
                K₂ = f(t[k + 1], η[k] .+ Δt.*(K₁ .+ ∫fₖ)) .- f(t[k + 1], η_old[k + 1])
                η[k + 1] = η[k] .+ 0.5Δt.*(K₁ .+ K₂) + Δt.*∫fₖ
            end
        end
    end

    return (t, η)
end


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


end