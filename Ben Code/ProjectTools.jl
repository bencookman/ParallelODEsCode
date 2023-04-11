"""
Key:
ODE -> Ordinary Differential equation
IVP -> Initial Value Problem
FE -> Forward Euler time-stepping scheme
RKp -> Runge-Kutta method of order p
IDC -> Integral Deferred Correction
RIDC -> Revisionist Integral Deferred Correction
RSDC -> Revisionist Spectral Deferred Correction
"""
module ProjectTools

using
    LinearAlgebra,
    Statistics,
    PyCall,
    FastGaussQuadrature,
    Parameters,
    BarycentricInterpolation,
    Elliptic.Jacobi

export
    ## Error calculation
    err_abs,
    err_rel,
    err_norm,
    err_rel_norm,
    err_max,
    ## Test IVP structure
    ODESystem,
    @unpack_ODESystem,
    ODETestSystem,
    @unpack_ODETestSystem,
    # Test IVPs
    Butcher_p53_system,
    sqrt_system,
    trig_system_vector,
    trig_system_scalar,
    poly_system_1,
    poly_system_2,
    poly_system_3,
    exp_system_1,
    exp_system_2,
    exp_system_3,
    log_system,
    Jacobi_system,
    ## Runge-Kutta methods
    RKMethod,
    @unpack_RKMethod,
    RK_time_step_explicit,
    RK4_standard,
    RK3_Kutta,
    RK2_midpoint,
    RK2_Heuns,
    RK1_forward_euler,
    RK8_Cooper_Verner,
    ## Integration matrix and polynomial interpolation
    integration_matrix,
    integration_matrix_Float32,
    integration_matrix_uniform,
    integration_matrix_uniform_RK4,
    integration_matrix_legendre,
    integration_matrix_legendre_RK4,
    integration_matrix_lobatto,
    integration_matrix_lobatto_RK4,
    INTEGRATION_MATRIX_ARRAY_UNIFORM,
    INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS,
    INTEGRATION_MATRIX_ARRAY_LEGENDRE,
    INTEGRATION_MATRIX_ARRAY_LOBATTO,
    INTEGRATION_MATRIX_ARRAY_LOBATTO_HALF_TIME_STEPS,
    fill_integration_matrix_array_uniform,
    fill_integration_matrix_array_uniform_half_time_steps,
    fill_integration_matrix_array_legendre,
    fill_integration_matrix_array_lobatto,
    fill_integration_matrix_array_lobatto_half_time_steps,
    interpolation_polynomials,
    interpolation_func,
    ## IDC ALGORITHMS
    # IDC
    IDC_FE_single,
    IDC_FE,
    IDC_RK2_Heuns,
    IDC_RK2_midpoint,
    IDC_RK3,
    IDC_RK4,
    # SDC
    sandwich_special_nodes,
    SDC_FE_legendre_single,
    SDC_FE_lobatto_single,
    SDC_FE_legendre,
    SDC_RK4_legendre,
    SDC_FE_lobatto,
    SDC_FE_lobatto_reduced_stencil,
    SDC_RK2_Heuns_lobatto,
    SDC_RK4_lobatto,
    # RIDC
    RIDC_FE,
    RIDC_FE_reduced_stencil,
    RIDC_RK2_Heuns_reduced_stencil,
    RIDC_RK4,
    # RSDC
    RSDC_FE_uniform,
    RSDC_RK4_uniform,
    RSDC_FE_lobatto,
    RSDC_FE_lobatto_reduced_stencil,
    RSDC_RK4_lobatto


### STUFF FOR TESTING

# Error metrics
err_abs(exact, approx) = abs.(exact .- approx)
err_rel(exact, approx) = err_abs(exact, approx) ./ abs.(exact)
err_norm(exact, approx, p) = norm(err_abs(exact, approx), p)
err_rel_norm(exact, approx, p) = err_norm(exact, approx, p)/norm(exact, p)
err_max(exact, approx) = maximum(err_abs(exact, approx))


"""
Structure to contain an ODE system with no exact solution
    f::Function  - gradient function, y' = f(t, y)
    t_s::Float64 - start time, y(t_s) is the solution at the initial time
    t_e::Float64 - end time, y(t_e) is the solution at the final time
    y_s::Float64 - starting value y(t_s) = y_s
"""
@with_kw struct ODESystem{T<:Function, S}
    f::T
    t_s::Float64
    t_e::Float64
    y_s::S
    ODESystem{T, S}(f, t_s, t_e, y_s) where {T<:Function, S} = (t_s < t_e) ? new(f, t_s, t_e, y_s) : error("time t_s must be less than time t_e")
end

# use form "Point(x::T, y::T) where {T<:Real} = Point{T}(x,y)"
ODESystem(f::T, t_s, t_e, y_s::S) where {T<:Function, S} = ODESystem{T, S}(f, t_s, t_e, y_s)
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

""" https://doi.org/10.1137/09075740X """
const f_trig(t, y) = begin
    one_minus_sqr_mag = 1 - y[1]^2 - y[2]^2
    [
        -y[2] +  y[1]*one_minus_sqr_mag,
         y[1] + 3y[2]*one_minus_sqr_mag
    ]
end
const y_trig(t) = [cos(t), sin(t)]
const trig_system_vector = ODETestSystem(
    f_trig,
    10.0,
    [1.0, 0.0],
    y_trig
)

""" https://doi.org/10.1090/S0025-5718-09-02276-5 """
const trig_system_scalar = ODETestSystem(
    (t, y) -> -2π*sin(2π*t) - 2*(y - cos(2π*t)),
    20.0,
    1.0,
    t -> cos(2π*t)
)

""" Problems with polynomial f should be 'easy' to solve numerically """
const poly_system_1 = ODETestSystem(
    (t, y) -> t^3,
    5.0,
    2.0,
    t -> 0.25*t^4 + 2.0,
)

const poly_system_2 = ODETestSystem(
    (t, y) -> (t - 10.0)^3,
    5.0,
    2.5e3,
    t -> 0.25*(t - 10.0)^4,
)

const poly_system_3 = ODETestSystem(
    (t, y) -> t^10,
    1.0,
    0.0,
    t -> (t^11)/11
)

const exp_system_1 = ODETestSystem(
    (t, y) -> 4y,
    3.0,
    1.0,
    t -> exp(4t)
)

""" simple example of a linear stiff system. """
const exp_system_2 = ODETestSystem(
    (t, y) -> -10y,
    3.0,
    1.0,
    t -> exp(-10t)
)

""" https://doi.org/10.2140/camcos.2009.4.27 """
const exp_system_3 = ODETestSystem(
    (t, y) -> y,
    1.0,
    1.0,
    t -> exp(t)
)

const log_system = ODETestSystem(
    (t, y) -> 0.5sin(t)*exp(y),
    π,
    log(2) - log(3),
    t -> -log(1 + 0.5cos(t))
)

""" https://doi.org/10.1023/A:1022338906936 """
const μ = 0.5
const f_Jacobi(t, y) = begin
    [
        y[2]*y[3],
        -y[1]*y[3],
        -μ*y[1]*y[2]
    ]
end
const y_Jacobi(t) = [sn(t, μ), cn(t, μ), dn(t, μ)]
const Jacobi_system = ODETestSystem(
    f_Jacobi,
    1.0,
    [0.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im],
    y_Jacobi
)



### RUNGE-KUTTA METHODS

"""
Represents an arbitrary Runge-Kutta method using its Butcher tableau. Contains:
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
        (length(axes(c)[1]) >= order_of_accuracy)  || error("system too small to support this order of accuracy")
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
        k_sum = zeros(eltype(u_i), size(u_i))
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
[REMOVED]
Perform a single explicit correction time-step step using the given Runge-Kutta scheme:

η^{l+1}_{i+1} = η^{l+1}ᵢ + Δt⋅∑bᵣ⋅kᵣ + Δt⋅Σ, where
kᵣ = f(tᵢ + cᵣ⋅Δt, η^{l+1}ᵢ + Δt⋅∑aᵣₗ⋅kₗ + cᵣ⋅Δt⋅Σ) - f(tᵢ + cᵣ⋅Δt, ηˡ(tᵢ + cᵣ⋅Δt))

( k = [k₁, k₂, ..., kₛ] )

η_old is the interpolant function η^{l+1}(t) evaluated at each time tᵢ + cᵣ⋅Δt.
"""
# function RK_correction_time_step_explicit(
#     f::Function,
#     Δt::Float64,
#     t_i::Float64,
#     η_i,
#     Σ,
#     η_old;
#     RK_method::RKMethod = RK4_standard
# )
#     @unpack_RKMethod RK_method

#     k = [f(t_i, η_i) - f(t_i, η_old[1])]
#     for r in axes(b)[1][2:end]
#         k_sum = zeros(typeof(η_i[1]), size(η_i))
#         for l in axes(b)[1][1]:(r - 1)
#             k_sum .+= a[r, l].*k[l]
#         end
#         k_next = f(t_i + c[r]*Δt, η_i .+ Δt.*k_sum .+ c[r].*Σ) - f(t_i + c[r]*Δt, η_old[r])
#         push!(k, k_next)
#     end
#     k_sum_full = zeros(typeof(η_i[1]), size(η_i))
#     for r in axes(b)[1]
#         k_sum_full .+= b[r].*k[r]
#     end
#     return Δt.*k_sum_full .+ Δt.*Σ
# end

""" Canonical RK4 method. """
const RK4_standard = RKMethod(
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
""" Kutta's third order Runge-Kutta method. """
const RK3_Kutta = RKMethod(
    a = [
        0//1  0//1  0//1;
        1//2  0//1  0//1;
        -1//1 2//1  0//1;
    ], b = [
        1//6, 2//3, 1//6
    ], c = [
        0//1, 1//2, 1//1
    ],
    order_of_accuracy = 3

)
""" Midpoint Runge-Kutta method. """
const RK2_midpoint = RKMethod(
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
const RK2_Heuns = RKMethod(
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
const RK1_forward_euler = RKMethod(
    a = fill(0, 1, 1),
    b = [1],
    c = [0],
    order_of_accuracy = 1
)
""" Eighth order Cooper-Verner method. """
const RK8_Cooper_Verner = RKMethod(
    a = Float64[
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.25 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        1/7 (-7 - 3sqrt(21))/98 (21 + 5sqrt(21))/49 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        (11 + sqrt(21))/84 0.0 (18 + 4sqrt(21))/63 (21 - sqrt(21))/252 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        (5 + sqrt(21))/48 0.0 (9 + sqrt(21))/36 (-231 + 14sqrt(21))/360 (63 - 7sqrt(21))/80 0.0 0.0 0.0 0.0 0.0 0.0;
        (10 - sqrt(21))/42 0.0 (-432 + 92sqrt(21))/315 (633 - 145sqrt(21))/90 (-504 + 115sqrt(21))/70 (63 - 13sqrt(21))/35 0.0 0.0 0.0 0.0 0.0;
        1/14 0.0 0.0 0.0 (14 - 3sqrt(21))/126 (13 - 3sqrt(21))/63 1/9 0.0 0.0 0.0 0.0;
        1/32 0.0 0.0 0.0 (91 - 21sqrt(21))/576 11/72 (-385 - 75sqrt(21))/1152 (63 + 13sqrt(21))/128 0.0 0.0 0.0;
        1/14 0.0 0.0 0.0 1/9 (-733 - 147sqrt(21))/2205 (515 + 111sqrt(21))/504 (-51 - 11sqrt(21))/56 (132 + 28sqrt(21))/245 0.0 0.0;
        0.0 0.0 0.0 0.0 (-42 + 7sqrt(21))/18 (-18 + 28sqrt(21))/45 (-273 - 53sqrt(21))/72 (301 + 53sqrt(21))/72 (28 - 28sqrt(21))/45 (49 - 7sqrt(21))/18 0.0
    ], b = Float64[
        1/20, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 49/180, 16/45,
        49/180, 1/20
    ], c = Float64[
        0.0, 0.5, 0.5,
        (7 + sqrt(21))/14, (7 + sqrt(21))/14, 0.5,
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
    η = initialise_η(y_s, N, 0)

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


### CALCULATING INTEGRATION MATRICES

"""
Switch between double (Float64) calculations of the integration matrix - this is
the default - and single (Float32) calculations of the integration matrix.

Used to test the impact of data precision on SDC integration effectiveness. It
turns out its very important due to catastrophic cancellation!
"""
function integration_matrix(t_quadrature_nodes, t_time_steps; use_double = true)
    S = use_double ?
        integration_matrix_Float64(collect(Float64, t_quadrature_nodes), collect(Float64, t_time_steps)) :
        integration_matrix_Float32(collect(Float32, t_quadrature_nodes), collect(Float32, t_time_steps))
    return S
end

""" """
function integration_matrix_Float64(t_quadrature_nodes::Vector{Float64}, t_time_steps::Vector{Float64})
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
    S = [
        qᵢ_func(t_time_steps[i + 1]) - qᵢ_func(t_time_steps[i])
        for i in axes(t_time_steps)[1][1:end - 1], qᵢ_func in q_func
    ]
    return S
end

function integration_matrix_Float32(t_quadrature_nodes::Vector{Float32}, t_time_steps::Vector{Float32})
    scipy_interpolate = pyimport("scipy.interpolate")
    # Calculate anti-derivatives of polynomial interpolants
    q_func = Function[]
    for i in axes(t_quadrature_nodes)[1]
        yᵢ = zeros(Float32, size(t_quadrature_nodes))
        yᵢ[i] = 1.0f0
        pᵢ_coeffs = scipy_interpolate.lagrange(t_quadrature_nodes, yᵢ).c |> reverse # Lagrange interpolant to yᵢ at t
        pᵢ_coeffs = Float32.(pᵢ_coeffs)
        qᵢ_coeffs = pᵢ_coeffs ./ (1.0f0:length(pᵢ_coeffs))                            # Antiderivative of pᵢ
        qᵢ_func(x::Float32) = sum(qᵢⱼ*x^j for (j, qᵢⱼ) in enumerate(qᵢ_coeffs))
        push!(q_func, qᵢ_func)
    end
    # Use antiderivatives to evaluate integration matrix
    S = [
        qᵢ_func(Float32(t_time_steps[i + 1])) - qᵢ_func(Float32(t_time_steps[i]))
        for i in axes(t_time_steps, 1)[1:end - 1], qᵢ_func in q_func
    ]
    return S
end

""" All of these are a function of the number of quadrature points, n """

integration_matrix_uniform(n; use_double = true) = integration_matrix(0.0:(n - 1), 0.0:(n - 1); use_double = use_double)

integration_matrix_uniform_RK4(n; use_double = true) = integration_matrix(0.0:(n - 1), 0.0:0.5:(n - 1); use_double = use_double)

function integration_matrix_legendre(n; use_double = true)
    legendre_nodes = gausslegendre(n)[1]
    return integration_matrix(legendre_nodes, vcat(-1, legendre_nodes, 1); use_double = use_double)
end

function integration_matrix_legendre_RK4(n; use_double = true)
    legendre_nodes = gausslegendre(n)[1]
    legendre_full_nodes = vcat(-1, legendre_nodes, 1)
    legendre_half_nodes = [legendre_full_nodes[1]]
    for i in axes(legendre_full_nodes, 1)[1:end - 1]
        half_node = 0.5*(legendre_full_nodes[i] + legendre_full_nodes[i + 1])
        push!(legendre_half_nodes, half_node)
        push!(legendre_half_nodes, legendre_full_nodes[i + 1])
    end

    return integration_matrix(legendre_nodes, legendre_half_nodes; use_double = use_double)
end

function integration_matrix_lobatto(n; use_double = true)
    lobatto_nodes = gausslobatto(n)[1]
    return integration_matrix(lobatto_nodes, lobatto_nodes; use_double = use_double)
end

function integration_matrix_lobatto_RK4(n; use_double = true)
    lobatto_nodes = gausslobatto(n)[1]
    lobatto_half_nodes = [lobatto_nodes[1]]
    for i in axes(lobatto_nodes, 1)[1:end - 1]
        half_node = 0.5*(lobatto_nodes[i] + lobatto_nodes[i + 1])
        push!(lobatto_half_nodes, half_node)
        push!(lobatto_half_nodes, lobatto_nodes[i + 1])
    end
    return integration_matrix(lobatto_nodes, lobatto_half_nodes; use_double = use_double)
end

"""
For cases where we are using multiple large weights matrices where we have a
large number of quadrature nodes, it is beneficial to have them precalculated. A
better solution would be to calculate these once and store them in an external
file (probably .csv for conveneience) somewhere, to be fetched as needed.

These matrices are ordered by the number of quadrature points they use. Notice,
the matrix for closed quadature nodes (uniform and lobatto) must start with a
single undefined element as there is no weights matrix defined for these using a
single node.
"""
const INTEGRATION_MATRIX_ARRAY_UNIFORM = Vector{Matrix{Float64}}(undef, 1)
const INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS = Vector{Matrix{Float64}}(undef, 1)
const INTEGRATION_MATRIX_ARRAY_LEGENDRE = Vector{Matrix{Float64}}(undef, 0)
const INTEGRATION_MATRIX_ARRAY_LOBATTO = Vector{Matrix{Float64}}(undef, 1)
const INTEGRATION_MATRIX_ARRAY_LOBATTO_HALF_TIME_STEPS = Vector{Matrix{Float64}}(undef, 1)

"""
Giving only maximum size as argument is reasonable as the matrices of smaller
size are significantly easier to generate, so they are of little cost. This also
means the above 'INTEGRATION_MATRIX_ARRAY_...'s are automatically ordered.
"""

function fill_integration_matrix_array_uniform(max_size; use_double = true)
    current_size = size(INTEGRATION_MATRIX_ARRAY_UNIFORM, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    number_quadrature_nodes_array = (current_size + 1):max_size
    push!(
        INTEGRATION_MATRIX_ARRAY_UNIFORM,
        [integration_matrix_uniform(number_q_nodes; use_double = use_double) for number_q_nodes in number_quadrature_nodes_array]...)
    nothing
end

"""
some IDC iterations require integration matrices generated over uniform nodes
witih half steps in between for stages (think midpoint method, RK4)
"""
function fill_integration_matrix_array_uniform_half_time_steps(max_size; use_double = true)
    current_size = size(INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    number_quadrature_nodes_array = (current_size + 1):max_size
    push!(
        INTEGRATION_MATRIX_ARRAY_UNIFORM_HALF_TIME_STEPS,
        [integration_matrix(1:number_q_nodes, 1:0.5:number_q_nodes; use_double = use_double) for number_q_nodes in number_quadrature_nodes_array]...)
    nothing
end

function fill_integration_matrix_array_legendre(max_size; use_double = true)
    current_size = size(INTEGRATION_MATRIX_ARRAY_LEGENDRE, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    number_quadrature_nodes_array = (current_size + 1):max_size
    push!(
        INTEGRATION_MATRIX_ARRAY_LEGENDRE,
        [integration_matrix_legendre(number_q_nodes; use_double = use_double) for number_q_nodes in number_quadrature_nodes_array]...)
    nothing
end

function fill_integration_matrix_array_lobatto(max_size; use_double = true)
    current_size = size(INTEGRATION_MATRIX_ARRAY_LOBATTO, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    number_quadrature_nodes_array = (current_size + 1):max_size
    push!(
        INTEGRATION_MATRIX_ARRAY_LOBATTO,
        [integration_matrix_lobatto(number_q_nodes; use_double = use_double) for number_q_nodes in number_quadrature_nodes_array]...)
    nothing
end

function fill_integration_matrix_array_lobatto_half_time_steps(max_size; use_double = true)
    current_size = size(INTEGRATION_MATRIX_ARRAY_LOBATTO_HALF_TIME_STEPS, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    number_quadrature_nodes_array = (current_size + 1):max_size
    push!(
        INTEGRATION_MATRIX_ARRAY_LOBATTO_HALF_TIME_STEPS,
        [integration_matrix_lobatto_RK4(number_q_nodes; use_double = use_double) for number_q_nodes in number_quadrature_nodes_array]...)
    nothing
end


""" assume range(t_values) = 1 """
function interpolation_polynomials(t_for_interp)
    scipy_interpolate = pyimport("scipy.interpolate")

    p_func = []
    for i in axes(t_for_interp)[1]
        yᵢ = zeros(Float64, size(t_for_interp))
        yᵢ[i] = 1.0
        pᵢ_coeffs = scipy_interpolate.lagrange(t_for_interp, yᵢ).c |> reverse # Lagrange interpolant to yᵢ at t
        pᵢ_func(x) = sum(coeff*x^(j - 1) for (j, coeff) in enumerate(pᵢ_coeffs))
        push!(p_func, pᵢ_func)
    end
    return p_func
end

function interpolation_func(η_values, p_func, t_start, t_end)
    f(t) = sum(η_values[i]*p_func[i](t) for i in axes(p_func, 1))
    return t -> f((t - t_start)/(t_end - t_start))
end

### IDC ALGORITHMS
"""
These are algorithms using the IDC framework.

Necessary arguments:
ODE_system:
    An ODE system without exact solution available. Contains f, [t_s, t_e] and y_s.
number_corrections:
    The number of IDC corrections to perform on top of the initial prediction.
S (or S_array when relevant):
    An integration matrix, contains quadrature weights for integration over a region
    [t_i, t_{i+1}].
(for RIDC) K:
    The number of time steps to perform before a 'restart'.

Potential arguments:
J:
    Number of groups IDC is performed on.
N:
    Total number of nodes to integrate over.
p:
    Desired order of approximation.
M:
    The number of subintervals per group.

Should return a tuple (t, η) where:
t:
    The time values where the solution to ODE_system has been approximated.
η:
    A matrix containing approximants at all correction levels

All functions should be suitable for non-scalar systems.
"""

function initialise_η_zeros(y_s, N, number_corrections)
    η = []
    if typeof(y_s) <: Number
        η = zeros(typeof(y_s), N + 1, number_corrections + 1)
    elseif typeof(y_s) <: Vector
        η = fill(zeros(eltype(y_s), size(y_s)[1]), N + 1, number_corrections + 1)
    end
    return η
end

function initialise_η(y_s, N, number_corrections)
    η = initialise_η_zeros(y_s, N, number_corrections)
    for level in 1:(number_corrections + 1)
        η[1, level] = y_s
    end
    return η
end

"""
Algorithm from https://doi.org/10.1137/09075740X
Note passing integration matrix S as argument is much more efficient for testing
as the same matrix may otherwise be calculated multiple times for different
function calls.
"""
function IDC_FE(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = number_corrections
    N = J*M
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1, 1] = η[k, 1] .+ Δt.*f(t[k], η[k, 1])
        end
        # Correction loop
        I = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I], η[I, level - 1])
            for m in 1:M
                k = j*M + m
                ∫fₘ = [sum(S[m, j]*f_group[j][i] for j in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end
                η[k + 1, level] = η[k, level] .+ Δt.*(f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])) .+ Δt.*∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

function IDC_RK2_Heuns(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = 2*(number_corrections + 1) - 1
    N = J*M
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k + 1], η[k, 1] .+ Δt.*stage_1)
            η[k + 1, 1] = η[k, 1] .+ 0.5Δt.*(stage_1 .+ stage_2)
        end
        # Correction loop
        I = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            for m in 1:M
                k = j*M + m
                f_group = f.(t[I], η[I, level - 1])
                ∫fₘ = [sum(S[m, j]*f_group[j][i] for j in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end
                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k + 1], η[k, level] .+ Δt.*stage_1 .+ Δt.*∫fₘ) .- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ 0.5Δt.*(stage_1 .+ stage_2) .+ Δt.*∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

function IDC_RK2_midpoint(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = 2*(number_corrections + 1) - 1
    N = J*M
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)
    interp_polys = interpolation_polynomials((0:M)./M)  # Get ̂π_j

    for j in 0:(J - 1)
        # RK2-midpoint prediction loop
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + 0.5Δt, η[k, 1] .+ 0.5Δt.*stage_1)
            η[k + 1, 1] = η[k, 1] .+ stage_2.*Δt
        end

        # RK2-midpoint corrections loop
        I_group = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                # Calculate approximate intetgrals over subintervals [t_m, t_m + 0.5Δt] etc.
                ∫fₘ_1 = Δt.*[sum(S[2*m - 1, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ_2 = Δt.*[sum(S[2*m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ = ∫fₘ_1 .+ ∫fₘ_2
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                    ∫fₘ_1 = ∫fₘ_1[1]
                end

                # Calulate η(t_m + 0.5Δt)
                η_prev_interp = interpolation_func(η[I_group, level - 1], interp_polys, t[I_group][1], t[I_group][end])
                η_prev_midₘ = η_prev_interp(t[k] + 0.5Δt)

                # Perform time step
                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k] + 0.5Δt, η[k, level] .+ 0.5Δt.*stage_1 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt, η_prev_midₘ)
                η[k + 1, level] = η[k, level] .+ Δt.*stage_2 .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

function IDC_RK3(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = 3*(number_corrections + 1) - 1
    N = J*M
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)
    interp_polys = interpolation_polynomials((0:M)./M)  # Get ̂π_j

    # RK3 prediction loop
    for j in 0:(J - 1)
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + 0.5Δt, η[k, 1] .+ 0.5Δt.*stage_1)
            stage_3 = f(t[k] + Δt, η[k, 1] .+ Δt.*(2.0.*stage_2 .- stage_1))
            η[k + 1, 1] = η[k, 1] .+ (stage_1 .+ 4.0.*stage_2 .+ stage_3).*Δt/6
        end

        # RK3 corrections loop
        I_group = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                # Calculate approximate intetgrals over subintervals [t_m, t_m + 0.5Δt] etc.
                ∫fₘ_1 = Δt.*[sum(S[2*m - 1, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ_2 = Δt.*[sum(S[2*m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ = ∫fₘ_1 .+ ∫fₘ_2
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                    ∫fₘ_1 = ∫fₘ_1[1]
                end

                # Calulate η(t_m + 0.5Δt)
                η_prev_interp = interpolation_func(η[I_group, level - 1], interp_polys, t[I_group][1], t[I_group][end])
                η_prev_midₘ = η_prev_interp(t[k] + 0.5Δt)

                # Perform time step
                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k] + 0.5Δt, η[k, level] .+ 0.5Δt.*stage_1 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt, η_prev_midₘ)
                stage_3 = f(t[k + 1], η[k, level] .+ Δt.*(2.0.*stage_2 .- stage_1) .+ ∫fₘ).- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ (Δt/6).*(stage_1 .+ 4.0.*stage_2 .+ stage_3) .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

function IDC_RK4(
    ODE_system::ODESystem,
    number_corrections, S, J, lagrange_basis_array
)
    @unpack_ODESystem ODE_system
    M = 4*(number_corrections + 1) - 1
    N = J*M
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    # Prediction loop
    for j in 0:(J - 1)
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + 0.5Δt, η[k, 1] .+ 0.5Δt.*stage_1)
            stage_3 = f(t[k] + 0.5Δt, η[k, 1] .+ 0.5Δt.*stage_2)
            stage_4 = f(t[k] + Δt, η[k, 1] .+ Δt.*stage_3)
            η[k + 1, 1] = η[k, 1] .+ (stage_1 .+ 2stage_2 .+ 2stage_3 .+ stage_4).*Δt/6
        end

        # Correction loop
        I_group = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                ∫fₘ_1 = Δt.*[sum(S[2*m - 1, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ_2 = Δt.*[sum(S[2*m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ = ∫fₘ_1 .+ ∫fₘ_2
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                    ∫fₘ_1 = ∫fₘ_1[1]
                end

                η_prev_interp = interpolation_func(η[I_group, level - 1], lagrange_basis_array, t[I_group][1], t[I_group][end])
                η_prev_midₘ = η_prev_interp(t[k] + 0.5Δt)

                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k] + 0.5Δt, η[k, level] .+ 0.5Δt.*stage_1 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt, η_prev_midₘ)
                stage_3 = f(t[k] + 0.5Δt, η[k, level] .+ 0.5Δt.*stage_2 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt, η_prev_midₘ)
                stage_4 = f(t[k + 1], η[k, level] .+ Δt.*stage_3 .+ ∫fₘ).- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ (Δt/6).*(stage_1 .+ stage_2.*2 .+ stage_3.*2 .+ stage_4) .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end


""" [REMOVED] not functional. """
# function IDC_RK_general(
#     ODE_system::ODESystem,
#     N, p, S;
#     RK_method::RKMethod = RK2_midpoint
# )
#     # Initialise variables
#     @unpack_ODESystem ODE_system
#     t = range(t_s, t_e, N + 1) |> collect
#     Δt = (t_e - t_s)/N
#     M = p - 1
#     J = fld(N, M)
#     η = zeros(typeof(y_s), N + 1)
#     η[1] = y_s

#     for j in 0:(J - 1)
#         # Prediction loop
#         for m in 1:M
#             k = j*M + m
#             η[k + 1] = RK_time_step_explicit(f, Δt, t[k], η[k]; RK_method = RK_method)
#         end
#         # Correction loop
#         I = (j*M + 1):((j + 1)*M + 1)   # Quadrature nodes
#         for _ in 2:fld(p, 2)
#             η_old = copy(η)
#             η_old_group_poly = get_IDC_RK_group_poly(η_old, t[j*M + 1], t[(j + 1)*M], (j*M + 1):((j + 1)*M))
#             for m in 1:M
#                 k = j*M + m
#                 ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
#                 η_old_interpolants = [η_old_group_poly(t[k] + Δt*cᵣ) for cᵣ in RK_method.c]
#                 η[k + 1] = RK_correction_time_step_explicit(
#                     f, Δt, t[k], η[k], ∫fₖ, η_old_interpolants;
#                     RK_method = RK_method
#                 )
#             end
#         end
#     end

#     return (t, η)
# end

"""
[NOT NEEDED]
Generates the polynomial which interpolates η_old at all points in a given group
and returns as a function of time.
"""
# function get_IDC_RK_group_poly(η_old, t_group_s, t_group_e, group_indices)
#     equispaced_poly = Equispaced{length(group_indices) - 1}()
#     interpolating_poly = interpolate(equispaced_poly, η_old[group_indices])
#     return (t_in -> interpolating_poly(2(t_in - t_group_s)/(t_group_e - t_group_s) - 1))
# end

"""
Integral deferred correction with a single subinterval (uses static 
interpolatory quadrature over all N + 1 nodes). To improve this could use a
reduced size integration matrix of minimum necessary size (p - 1 = M) with
rolling usage of interpolation nodes.
"""
function IDC_FE_single(
    ODE_system::ODESystem,
    number_corrections, S, N
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    # Prediction loop
    for m in 1:N
        η[m + 1, 1] = η[m, 1] .+ Δt.*f(t[m], η[m, 1])
    end
    # Correction loop
    I = 1:(N + 1)
    for level in 2:(number_corrections + 1)
        f_group = f.(t[I], η[I, level - 1])
        for m in 1:N
            ∫fₘ = [sum(S[m, j]*f_group[j][i] for j in axes(S, 2)) for i in axes(y_s, 1)]
            if typeof(y_s) <: Number
                ∫fₘ = ∫fₘ[1]
            end
            η[m + 1, level] = η[m, level] .+ Δt.*(f(t[m], η[m, level]) .- f(t[m], η[m, level - 1])) .+ Δt.*∫fₘ
        end
    end

    return (t, η)
end


## NON-UNIFORM NODES

function sandwich_special_nodes(
    special_nodes, is_closed::Bool, t_s, t_e, J
)
    t_return = []
    group_size = (t_e - t_s)/J
    group_scale_factor = group_size/2
    if is_closed
        for j in 0:(J - 1)
            current_group_nodes = (special_nodes[1:end - 1] .- special_nodes[1]).*group_scale_factor .+ j*group_size
            push!(t_return, current_group_nodes...)
        end
    else
        for j in 0:(J - 1)
            push!(t_return, j*group_size)
            current_group_nodes = (special_nodes .+ 1).*group_scale_factor .+ j*group_size
            push!(t_return, current_group_nodes...)
        end
    end
    push!(t_return, t_e)

    return (t_return, group_scale_factor)
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
"""
function SDC_FE_legendre_single(
    ODE_system::ODESystem,
    number_corrections, S, N
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t_legendre = gausslegendre(N - 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(t_legendre, false, t_s, t_e, 1)
    Δt = t[2:end] .- t[1:end - 1]   # No longer constant
    η = initialise_η(y_s, N, number_corrections)

    # Prediction loop
    for m in 1:N
        η[m + 1, 1] = η[m, 1] .+ Δt[m].*f(t[m], η[m, 1])
    end
    # Correction loop
    I = 2:N
    for level in 2:(number_corrections + 1)
        f_group = f.(t[I], η[I, level - 1])
        for m in 1:N
            ∫fₘ = group_scale_factor.*[sum(S[m, j]*f_group[j][i] for j in axes(S, 2)) for i in axes(y_s, 1)]
            if typeof(y_s) <: Number
                ∫fₘ = ∫fₘ[1]
            end
            η[m + 1, level] = η[m, level] .+ Δt[m].*(f(t[m], η[m, level]) .- f(t[m], η[m, level - 1])) .+ ∫fₘ
        end
    end

    return (t, η)
end

function SDC_FE_lobatto_single(
    ODE_system::ODESystem,
    number_corrections, S, N
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    t_lobatto = gausslobatto(N + 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(t_lobatto, true, t_s, t_e, 1)
    Δt = t[2:end] .- t[1:end - 1]   # No longer constant
    η = initialise_η(y_s, N, number_corrections)

    # Prediction loop
    for m in 1:N
        η[m + 1, 1] = η[m, 1] .+ Δt[m].*f(t[m], η[m, 1])
    end
    # Correction loop
    I = 1:(N + 1)
    for level in 2:(number_corrections + 1)
        f_group = f.(t[I], η[I, level - 1])
        for m in 1:N
            ∫fₘ = group_scale_factor.*[sum(S[m, j]*f_group[j][i] for j in axes(S, 2)) for i in axes(y_s, 1)]
            if typeof(y_s) <: Number
                ∫fₘ = ∫fₘ[1]
            end
            η[m + 1, level] = η[m, level] .+ Δt[m].*(f(t[m], η[m, level]) .- f(t[m], η[m, level - 1])) .+ ∫fₘ
        end
    end

    return (t, η)
end


function SDC_FE_legendre(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = number_corrections + 2
    N = J*M
    legendre_nodes = gausslegendre(number_corrections + 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(legendre_nodes, false, t_s, t_e, J)
    Δt = t[2:end] .- t[1:end - 1]   # No longer constant
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1, 1] = η[k, 1] .+ Δt[k].*f(t[k], η[k, 1])
        end
        # Correction loop
        I = (j*M + 2):((j + 1)*M)   # Use non-endpoint nodes for quadrature
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I], η[I, level - 1])
            for m in 1:M
                k = j*M + m
                ∫fₘ = [sum(S[m, j]*f_group[j][i] for j in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end
                η[k + 1, level] = η[k, level] .+ Δt[k].*(f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])) .+ ∫fₘ.*group_scale_factor
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

"""
By
https://doi.org/10.1090/S0025-5718-09-02276-5
this algorithm should not achieve the desired order 4*(number_corrections + 1)
"""
function SDC_RK4_legendre(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    @unpack_ODESystem ODE_system
    M = 4*(number_corrections + 1) + 1
    N = J*M
    legendre_nodes = gausslegendre(M - 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(legendre_nodes, false, t_s, t_e, J)
    Δt = t[2:end] .- t[1:end - 1]
    η = initialise_η(y_s, N, number_corrections)
    interp_polys = interpolation_polynomials((legendre_nodes .+ 1)./2)

    for j in 0:(J - 1)
        # Predict
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + 0.5Δt[k], η[k, 1] .+ 0.5Δt[k].*stage_1)
            stage_3 = f(t[k] + 0.5Δt[k], η[k, 1] .+ 0.5Δt[k].*stage_2)
            stage_4 = f(t[k] + Δt[k], η[k, 1] .+ Δt[k].*stage_3)
            η[k + 1, 1] = η[k, 1] .+ (stage_1 .+ 2stage_2 .+ 2stage_3 .+ stage_4).*Δt[k]/6
        end

        # Correct
        I_prev_quad = (j*M + 2):((j + 1)*M)     # Use non-endpoint nodes for quadrature
        for level in 2:(number_corrections + 1)
            f_prev_quad = f.(t[I_prev_quad], η[I_prev_quad, level - 1])
            for m in 1:M
                k = j*M + m

                ∫fₘ_1 = group_scale_factor.*[sum(S[2*m - 1, s]*f_prev_quad[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ_2 = group_scale_factor.*[sum(S[2*m, s]*f_prev_quad[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ = ∫fₘ_1 .+ ∫fₘ_2
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                    ∫fₘ_1 = ∫fₘ_1[1]
                end

                η_prev_interp = interpolation_func(η[I_prev_quad, level - 1], interp_polys, t[I_prev_quad[1] - 1], t[I_prev_quad[end] + 1])
                η_prev_midₘ = η_prev_interp(t[k] + 0.5Δt[k])

                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k] + 0.5Δt[k], η[k, level] .+ 0.5Δt[k].*stage_1 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt[k], η_prev_midₘ)
                stage_3 = f(t[k] + 0.5Δt[k], η[k, level] .+ 0.5Δt[k].*stage_2 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt[k], η_prev_midₘ)
                stage_4 = f(t[k + 1], η[k, level] .+ Δt[k].*stage_3 .+ ∫fₘ).- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ (Δt[k]/6).*(stage_1 .+ stage_2.*2 .+ stage_3.*2 .+ stage_4) .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

function SDC_FE_lobatto(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    @unpack_ODESystem ODE_system
    M = number_corrections
    N = J*M
    t_gl = gausslobatto(M + 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(t_gl, true, t_s, t_e, J)
    Δt = t[2:end] .- t[1:end - 1]
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Predict
        for m in 1:M
            k = j*M + m
            η[k + 1, 1] = η[k, 1] .+ f(t[k], η[k, 1]).*Δt[k]
        end

        # Correct
        I_group = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                ∫fₘ = group_scale_factor.*[sum(S[m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end

                η[k + 1, level] = η[k, level] .+ Δt[k].*(f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])) .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

function SDC_FE_lobatto_reduced_stencil(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    @unpack_ODESystem ODE_system
    M = ceil(Int64, (number_corrections + 1)/2)
    N = J*M
    t_gl = gausslobatto(M + 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(t_gl, true, t_s, t_e, J)
    Δt = t[2:end] .- t[1:end - 1]
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Predict
        for m in 1:M
            k = j*M + m
            η[k + 1, 1] = η[k, 1] .+ f(t[k], η[k, 1]).*Δt[k]
        end

        # Correct
        I_group = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                ∫fₘ = group_scale_factor.*[sum(S[m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end

                η[k + 1, level] = η[k, level] .+ Δt[k].*(f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])) .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

"""
By
https://doi.org/10.1090/S0025-5718-09-02276-5
this algorithm should not work.
"""
function SDC_RK2_Heuns_lobatto(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    @unpack_ODESystem ODE_system
    M = 2*(number_corrections + 1) - 1
    N = J*M
    t_gl = gausslobatto(M + 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(t_gl, true, t_s, t_e, J)
    Δt = t[2:end] .- t[1:end - 1]
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Predict
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + Δt[k], η[k, 1] .+ Δt[k].*stage_1)
            η[k + 1, 1] = η[k, 1] .+ (stage_1 .+ stage_2).*Δt[k]/2
        end

        # Correct
        I_group = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                ∫fₘ = group_scale_factor.*[sum(S[m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end

                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k + 1], η[k, level] .+ Δt[k].*stage_1 .+ ∫fₘ).- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ (Δt[k]/2).*(stage_1 .+ stage_2) .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

"""
By
https://doi.org/10.1090/S0025-5718-09-02276-5
this algorithm should not work.
"""
function SDC_RK4_lobatto(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    @unpack_ODESystem ODE_system
    M = 4*(number_corrections + 1) - 1
    N = J*M
    t_gl = gausslobatto(M + 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(t_gl, true, t_s, t_e, J)
    Δt = t[2:end] .- t[1:end - 1]
    η = initialise_η(y_s, N, number_corrections)
    interp_polys = interpolation_polynomials((t_gl .+ 1)./2)

    for j in 0:(J - 1)
        # Predict
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + 0.5Δt[k], η[k, 1] .+ 0.5Δt[k].*stage_1)
            stage_3 = f(t[k] + 0.5Δt[k], η[k, 1] .+ 0.5Δt[k].*stage_2)
            stage_4 = f(t[k] + Δt[k], η[k, 1] .+ Δt[k].*stage_3)
            η[k + 1, 1] = η[k, 1] .+ (stage_1 .+ 2stage_2 .+ 2stage_3 .+ stage_4).*Δt[k]/6
        end

        # Correct
        I_group = (j*M + 1):((j + 1)*M + 1)
        for level in 2:(number_corrections + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                ∫fₘ_1 = group_scale_factor.*[sum(S[2*m - 1, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ_2 = group_scale_factor.*[sum(S[2*m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ = ∫fₘ_1 .+ ∫fₘ_2
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                    ∫fₘ_1 = ∫fₘ_1[1]
                end

                η_prev_interp = interpolation_func(η[I_group, level - 1], interp_polys, t[I_group][1], t[I_group][end])
                η_prev_midₘ = η_prev_interp(t[k] + 0.5Δt[k])

                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k] + 0.5Δt[k], η[k, level] .+ 0.5Δt[k].*stage_1 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt[k], η_prev_midₘ)
                stage_3 = f(t[k] + 0.5Δt[k], η[k, level] .+ 0.5Δt[k].*stage_2 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt[k], η_prev_midₘ)
                stage_4 = f(t[k + 1], η[k, level] .+ Δt[k].*stage_3 .+ ∫fₘ).- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ (Δt[k]/6).*(stage_1 .+ stage_2.*2 .+ stage_3.*2 .+ stage_4) .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*M + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end



## RIDC

"""
The sequential form of our RIDC algorithm is relatively simple: we've simply
added more space in each subinterval whilst interpolating over the same number
of nodes. The main drawback to this potential parallelisability is that only the
less stable uniform nodes may be used.
"""
function RIDC_FE(
    ODE_system::ODESystem,
    number_corrections, S, J, K
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = number_corrections
    N = J*K
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:K
            k = j*K + m
            η[k + 1, 1] = η[k, 1] .+ Δt.*f(t[k], η[k, 1])
        end
        # Correction loop
        for level in 2:(number_corrections + 1)
            I_group = (j*K + 1):((j + 1)*K + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:K
                k = j*K + m
                is_quad_startup = (m in 1:M)
                quad_start = is_quad_startup ? 1 : m + 1 - M
                integ_interval = is_quad_startup ? m : M
                ∫fₘ = [sum(S[integ_interval, s]*f_group[quad_start + s - 1][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end
                η[k + 1, level] = η[k, level] .+ Δt.*(f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])) .+ Δt.*∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*K + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

"""
For forward Euler, at each correction step we are only 'aiming' for an order
equal to the number of corrections previously done (plus 2). In lieu of this, we
only require interpolatory quadrature involving a number of nodes again related
to # of previous correction steps.
"""
function RIDC_FE_reduced_stencil(
    ODE_system::ODESystem,
    number_corrections, S_array, J, K
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = number_corrections
    N = J*K
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:K
            k = j*K + m
            η[k + 1, 1] = η[k, 1] .+ Δt.*f(t[k], η[k, 1])
        end
        # Correction loop
        for level in 2:(number_corrections + 1)
            I_group = (j*K + 1):((j + 1)*K + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:K
                k = j*K + m
                is_quad_startup = (m in 1:level - 1)
                quad_start = is_quad_startup ? 1 : m + 2 - level
                integ_interval = is_quad_startup ? m : level - 1
                ∫fₘ = [
                    sum(S_array[level - 1][integ_interval, s]*f_group[quad_start + s - 1][i]
                    for s in axes(S_array[level - 1], 2)) for i in axes(y_s, 1)
                ]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end
                η[k + 1, level] = η[k, level] .+ Δt.*(f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])) .+ Δt.*∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*K + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

"""  """
function RIDC_RK2_Heuns_reduced_stencil(
    ODE_system::ODESystem,
    number_corrections, S_levels, J, K
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M_levels = [2*level - 1 for level in 2:(number_corrections + 1)]
    N = J*K
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:K
            k = j*K + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k + 1], η[k, 1] .+ Δt.*stage_1)
            η[k + 1, 1] = η[k, 1] .+ 0.5Δt.*(stage_1 .+ stage_2)
        end

        # Correction loop
        for level in 2:(number_corrections + 1)
            I_group = (j*K + 1):((j + 1)*K + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:K
                k = j*K + m

                is_quad_startup = (m in 1:M_levels[level - 1])
                quad_start = is_quad_startup ? 1 : m + 1 - M_levels[level - 1]
                integ_interval = is_quad_startup ? m : M_levels[level - 1]
                ∫fₘ = [
                    sum(
                        S_levels[level - 1][integ_interval, s]*f_group[quad_start + s - 1][i]
                        for s in axes(S_levels[level - 1], 2)
                    )
                    for i in axes(y_s, 1)
                ]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end

                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k + 1], η[k, level] .+ Δt.*stage_1 .+ Δt.*∫fₘ) .- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ 0.5Δt.*(stage_1 .+ stage_2) .+ Δt.*∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*K + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end

function RIDC_RK4(
    ODE_system::ODESystem,
    number_corrections, S, J, K
)
    @unpack_ODESystem ODE_system
    M = 4*(number_corrections + 1) - 1
    N = J*K
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    interp_polys = interpolation_polynomials((0:M)./M)

    # Prediction loop
    for j in 0:(J - 1)
        for m in 1:K
            k = j*K + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + 0.5Δt, η[k, 1] .+ 0.5Δt.*stage_1)
            stage_3 = f(t[k] + 0.5Δt, η[k, 1] .+ 0.5Δt.*stage_2)
            stage_4 = f(t[k] + Δt, η[k, 1] .+ Δt.*stage_3)
            η[k + 1, 1] = η[k, 1] .+ (stage_1 .+ 2stage_2 .+ 2stage_3 .+ stage_4).*Δt/6
        end

        # Correction loop
        for level in 2:(number_corrections + 1)
            I_group = (j*K + 1):((j + 1)*K + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:K
                k = j*K + m

                is_quad_startup = (m in 1:M)
                quad_start = is_quad_startup ? 1 : m + 1 - M
                integ_interval = is_quad_startup ? m : M
                ∫fₘ_1 = Δt.*[
                    sum(S[2*integ_interval - 1, s]*f_group[quad_start + s - 1][i] for s in axes(S, 2))
                    for i in axes(y_s, 1)
                ]
                ∫fₘ_2 = Δt.*[
                    sum(S[2*integ_interval, s]*f_group[quad_start + s - 1][i] for s in axes(S, 2))
                    for i in axes(y_s, 1)
                ]
                ∫fₘ = ∫fₘ_1 .+ ∫fₘ_2
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                    ∫fₘ_1 = ∫fₘ_1[1]
                end

                I = (j*K + quad_start):(j*K + quad_start + M)
                η_prev_interp = interpolation_func(η[I, level - 1], interp_polys, t[I][1], t[I][end])
                η_prev_midₘ = η_prev_interp(t[k] + 0.5Δt)

                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k] + 0.5Δt, η[k, level] .+ 0.5Δt.*stage_1 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt, η_prev_midₘ)
                stage_3 = f(t[k] + 0.5Δt, η[k, level] .+ 0.5Δt.*stage_2 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt, η_prev_midₘ)
                stage_4 = f(t[k + 1], η[k, level] .+ Δt.*stage_3 .+ ∫fₘ).- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ (Δt/6).*(stage_1 .+ stage_2.*2 .+ stage_3.*2 .+ stage_4) .+ ∫fₘ
            end
        end
        # At the end of group evaluation, give most corrected value to all nodes
        # for next group's evaluation.
        # (j == J - 1) && continue
        k_end = (j + 1)*K + 1
        for level in 1:number_corrections
            η[k_end, level] = η[k_end, end]
        end
    end

    return (t, η)
end



## IDC ACROSS GROUPS

function predict_group(
    η_start, t_group,
    f, value_type,
    M, Δt
)
    η_group = zeros(value_type, M + 1)
    η_group[1] = η_start
    for m in 1:M
        η_group[m + 1] = η_group[m] + Δt*f(t_group[m], η_group[m])
    end
    return η_group
end

function correct_group(
    η_start, t_group, η_old_group,
    f, value_type,
    M, S, Δt
)
    η_group = zeros(value_type, M + 1)
    η_group[1] = η_start
    for m in 1:M
        ∫fₘ = dot(S[m, :], f.(t_group, η_old_group))
        η_group[m + 1] = η_group[m] + Δt*(f(t_group[m], η_group[m]) - f(t_group[m], η_old_group[m])) + Δt*∫fₘ
    end
    return η_group
end

"""
A slightly modified IDC method where we do the prediction and correction levels
over the whole time domain sequentially. This allows for parallelisation 'over
the groups'.

With uniform nodes this is effectively RIDC(K = N, J = 1).
"""
function RSDC_FE_uniform(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    # Initialise variables
    @unpack_ODESystem ODE_system
    M = number_corrections
    N = J*M
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1, 1] = η[k, 1] .+ Δt.*f(t[k], η[k, 1])
        end
    end
        # Correction loop
    for level in 2:(number_corrections + 1)
        for j in 0:(J - 1)
            I = (j*M + 1):((j + 1)*M + 1)
            f_group = f.(t[I], η[I, level - 1])
            for m in 1:M
                k = j*M + m
                ∫fₘ = [sum(S[m, j]*f_group[j][i] for j in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end
                η[k + 1, level] = η[k, level] .+ Δt.*(f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])) .+ Δt.*∫fₘ
            end
        end
    end

    return (t, η)
end

function RSDC_RK4_uniform(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    @unpack_ODESystem ODE_system
    M = 4*(number_corrections + 1) - 1
    N = J*M
    t = range(t_s, t_e, N + 1)
    Δt = (t_e - t_s)/N
    η = initialise_η(y_s, N, number_corrections)
    interp_polys = interpolation_polynomials((0:M)./M)

    # Prediction loop
    for j in 0:(J - 1)
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + 0.5Δt, η[k, 1] .+ 0.5Δt.*stage_1)
            stage_3 = f(t[k] + 0.5Δt, η[k, 1] .+ 0.5Δt.*stage_2)
            stage_4 = f(t[k] + Δt, η[k, 1] .+ Δt.*stage_3)
            η[k + 1, 1] = η[k, 1] .+ (stage_1 .+ 2stage_2 .+ 2stage_3 .+ stage_4).*Δt/6
        end
    end

    # Correction loop
    for level in 2:(number_corrections + 1)
        for j in 0:(J - 1)
            I_group = (j*M + 1):((j + 1)*M + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                ∫fₘ_1 = Δt.*[sum(S[2*m - 1, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ_2 = Δt.*[sum(S[2*m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ = ∫fₘ_1 .+ ∫fₘ_2
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                    ∫fₘ_1 = ∫fₘ_1[1]
                end

                η_prev_interp = interpolation_func(η[I_group, level - 1], interp_polys, t[I_group][1], t[I_group][end])
                η_prev_midₘ = η_prev_interp(t[k] + 0.5Δt)

                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k] + 0.5Δt, η[k, level] .+ 0.5Δt.*stage_1 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt, η_prev_midₘ)
                stage_3 = f(t[k] + 0.5Δt, η[k, level] .+ 0.5Δt.*stage_2 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt, η_prev_midₘ)
                stage_4 = f(t[k + 1], η[k, level] .+ Δt.*stage_3 .+ ∫fₘ).- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ (Δt/6).*(stage_1 .+ stage_2.*2 .+ stage_3.*2 .+ stage_4) .+ ∫fₘ
            end
        end
    end

    return (t, η)
end

function RSDC_FE_lobatto(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    @unpack_ODESystem ODE_system
    M = number_corrections
    N = J*M
    t_gl = gausslobatto(M + 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(t_gl, true, t_s, t_e, J)
    Δt = t[2:end] .- t[1:end - 1]
    η = initialise_η(y_s, N, number_corrections)

    # Predict over groups
    for j in 0:(J - 1)
        for m in 1:M
            k = j*M + m
            η[k + 1, 1] = η[k, 1] .+ Δt[k].*f(t[k], η[k, 1])
        end
    end

    # Correct over groups
    for level in 2:(number_corrections + 1)
        for j in 0:(J - 1)
            I = (j*M + 1):((j + 1)*M + 1)
            f_group = f.(t[I], η[I, level - 1])
            for m in 1:M
                k = j*M + m
                ∫fₘ = [sum(S[m, j]*f_group[j][i] for j in axes(S, 2)) for i in axes(y_s, 1)]
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                end
                η[k + 1, level] = η[k, level] .+ Δt[k].*(f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])) .+ group_scale_factor.*∫fₘ
            end
        end
    end

    return (t, η)
end

function RSDC_FE_lobatto_reduced_stencil(
    ODE_system::ODESystem,
    J, number_corrections, S_levels, number_final_level_nodes
)
    scipy_bary_interp = pyimport("scipy.interpolate").BarycentricInterpolator

    @unpack_ODESystem ODE_system
    p = number_corrections + 1

    # Each level does a different number of approximations
    η = [zeros(typeof(y_s), level*J + 1) for level in 1:number_corrections]
    push!(η, zeros(typeof(y_s), number_final_level_nodes*J + 1))
    for level in 1:p
        η[level][1] = y_s
    end

    t_groups = range(t_s, t_e, J + 1)
    Δt_group = t_groups[2] - t_groups[1]
    # All necessary Gauss-Lobatto nodes over [-1, 1] and scale these to nodes over [0, Δt_group]
    t_gl_levels = [gausslobatto(1 + level)[1] for level in 1:number_corrections]
    push!(t_gl_levels, gausslobatto(1 + number_final_level_nodes)[1])
    for (level, t_gl) in enumerate(t_gl_levels)
        t_gl = (t_gl .+ 1).*(Δt_group/2)
        t_gl_levels[level] = t_gl
    end

    (number_corrections == length(S_levels)) || throw(error("there must be as many integration matrices as corrections"))
    (number_final_level_nodes >= 1) || throw(error("must have at least one node in the final level"))

    # Predict over groups
    for j in 0:(J - 1)
        k = j + 1
        t_gl_current = t_gl_levels[1]
        η[1][k + 1] = η[1][k] + Δt_group*f(t_gl_current[1] + t_groups[j + 1], η[1][k])
    end

    # Correct over groups
    for level in 2:p
        t_gl_current = t_gl_levels[level]
        t_gl_prev = t_gl_levels[level - 1]
        Δt_gl_current = t_gl_current[2:end] .- t_gl_current[1:end - 1]
        for j in 0:(J - 1)
            t_current = t_gl_current .+ t_groups[j + 1]
            t_prev = t_gl_prev .+ t_groups[j + 1]

            group_indices = (j*(length(t_prev) - 1) + 1):((j + 1)*(length(t_prev) - 1) + 1)
            # Calculate interpolating polynomial over previous level's group of values
            η_prev_interp = scipy_bary_interp(t_prev, η[level - 1][group_indices])
            for m in 1:(length(t_current) - 1)
                k = j*(length(t_current) - 1) + m

                ∫fₘ = dot(S_levels[level - 1][m, :], f.(t_prev, η[level - 1][group_indices]))*Δt_group/2
                η_prevₘ = η_prev_interp(t_current[m])[1]
                η[level][k + 1] = η[level][k] + Δt_gl_current[m]*(f(t_current[m], η[level][k]) - f(t_current[m], η_prevₘ)) + ∫fₘ
            end
        end
    end

    return (t_groups, η)
end

"""
By
https://doi.org/10.1090/S0025-5718-09-02276-5
this algorithm should not work.
"""
function RSDC_RK4_lobatto(
    ODE_system::ODESystem,
    number_corrections, S, J
)
    @unpack_ODESystem ODE_system
    M = 4*(number_corrections + 1) - 1
    N = J*M
    t_gl = gausslobatto(M + 1)[1]
    (t, group_scale_factor) = sandwich_special_nodes(t_gl, true, t_s, t_e, J)
    Δt = t[2:end] .- t[1:end - 1]
    η = initialise_η(y_s, N, number_corrections)
    interp_polys = interpolation_polynomials((t_gl .+ 1)./2)

    # Predict over groups
    for j in 0:(J - 1)
        for m in 1:M
            k = j*M + m
            stage_1 = f(t[k], η[k, 1])
            stage_2 = f(t[k] + 0.5Δt[k], η[k, 1] .+ 0.5Δt[k].*stage_1)
            stage_3 = f(t[k] + 0.5Δt[k], η[k, 1] .+ 0.5Δt[k].*stage_2)
            stage_4 = f(t[k] + Δt[k], η[k, 1] .+ Δt[k].*stage_3)
            η[k + 1, 1] = η[k, 1] .+ (stage_1 .+ 2stage_2 .+ 2stage_3 .+ stage_4).*Δt[k]/6
        end
    end

    # Correct over groups
    for level in 2:(number_corrections + 1)
        for j in 0:(J - 1)
            I_group = (j*M + 1):((j + 1)*M + 1)
            f_group = f.(t[I_group], η[I_group, level - 1])
            for m in 1:M
                k = j*M + m

                ∫fₘ_1 = group_scale_factor.*[sum(S[2*m - 1, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ_2 = group_scale_factor.*[sum(S[2*m, s]*f_group[s][i] for s in axes(S, 2)) for i in axes(y_s, 1)]
                ∫fₘ = ∫fₘ_1 .+ ∫fₘ_2
                if typeof(y_s) <: Number
                    ∫fₘ = ∫fₘ[1]
                    ∫fₘ_1 = ∫fₘ_1[1]
                end

                η_prev_interp = interpolation_func(η[I_group, level - 1], interp_polys, t[I_group][1], t[I_group][end])
                η_prev_midₘ = η_prev_interp(t[k] + 0.5Δt[k])

                stage_1 = f(t[k], η[k, level]) .- f(t[k], η[k, level - 1])
                stage_2 = f(t[k] + 0.5Δt[k], η[k, level] .+ 0.5Δt[k].*stage_1 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt[k], η_prev_midₘ)
                stage_3 = f(t[k] + 0.5Δt[k], η[k, level] .+ 0.5Δt[k].*stage_2 .+ ∫fₘ_1) .- f(t[k] + 0.5Δt[k], η_prev_midₘ)
                stage_4 = f(t[k + 1], η[k, level] .+ Δt[k].*stage_3 .+ ∫fₘ).- f(t[k + 1], η[k + 1, level - 1])
                η[k + 1, level] = η[k, level] .+ (Δt[k]/6).*(stage_1 .+ stage_2.*2 .+ stage_3.*2 .+ stage_4) .+ ∫fₘ
            end
        end
    end

    return (t, η)
end



end