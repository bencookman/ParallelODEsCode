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

    RKMethod,
    @unpack_RKMethod,
    RK_time_step_explicit,
    RK4_standard,
    RK2_midpoint,
    RK2_trapezoid,
    RK1_forward_euler,

    integration_matrix,
    integration_matrix_equispaced,
    integration_matrix_legendre,

    IDC_FE,
    IDC_FE_correction_levels,
    IDC_RK2,
    IDC_RK_general,
    get_IDC_RK_group_poly,
    IDC_FE_single,
    IDC_FE_single_correction_levels,
    SDC_FE,
    SDC_FE_single,
    RIDC_FE_sequential,
    RIDC_FE_sequential_correction_levels,
    RIDC_FE_sequential_reduced_stencil

err_abs(exact, approx) = abs.(exact - approx)
err_rel(exact, approx) = err_abs(exact, approx) ./ abs.(exact)
err_cum(exact, approx) = [sum(err_abs(exact[1:i], approx[1:i])) for i in 1:length(data_correct)]
err_norm(exact, approx, p) = norm(err_abs(exact, approx), p)
err_max(exact, approx)  = maximum(err_abs(exact, approx))

"""
Structure to contain a test ODE system. Contains:
    f::Function  - gradient function, y' = f(t, y)
    y::Function  - exact solution
    y_s::Float64 - starting value y(t_s) = y_s
    t_s::Float64 - start time, y(t_s) is the solution at the initial time
    t_e::Float64 - end time, y(t_e) is the solution at the final time
"""
@with_kw struct ODESystem
    f::Function
    y::Function
    y_s::Float64
    t_s::Float64
    t_e::Float64
    ODESystem(f, y, y_s, t_e) = new(f, y, y_s, 0, t_e)
end

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
    b = [0],
    c = [1],
    order_of_accuracy = 1
)


function integration_matrix(t::Array{T}) where T
    scipy_interpolate = pyimport("scipy.interpolate")

    N = length(t) - 1
    # Calculate anti-derivatives of polynomial interpolants
    q_func = []
    for i in 1:(N + 1)
        yᵢ = zeros(Float64, N + 1)
        yᵢ[i] = 1.0
        pᵢ_coeffs = scipy_interpolate.lagrange(t, yᵢ).c |> reverse              # Lagrange interpolant to yᵢ at x
        qᵢ_coeffs = pᵢ_coeffs ./ collect(1.0:length(pᵢ_coeffs))                 # Anti-derivative of pᵢ
        qᵢ_func(x) = sum(qᵢⱼ*x^Float64(j) for (j, qᵢⱼ) in enumerate(qᵢ_coeffs))
        push!(q_func, qᵢ_func)
    end
    # Use anti-derivatives to evaluate integration matrix
    return [qᵢ_func(t[i + 1]) - qᵢ_func(t[i]) for i in 1:N, qᵢ_func in q_func]
end

function integration_matrix_equispaced(N::Int)
    t = 0.0:N |> collect
    return integration_matrix(t)
end

function integration_matrix_legendre(N::Int, t_part::Array{T}) where T
    scipy_interpolate = pyimport("scipy.interpolate")

    # Calculate anti-derivatives of polynomial interpolants
    q_func = []
    gl_quad = gausslegendre(N)
    t_quad = gl_quad[1]              # Legendre nodes in [-1, 1]
    for i in 1:N
        yᵢ = zeros(Float64, N)
        yᵢ[i] = 1.0
        pᵢ_coeffs = scipy_interpolate.lagrange(t_quad, yᵢ).c |> reverse              # Lagrange interpolant to yᵢ at x
        qᵢ_coeffs = pᵢ_coeffs ./ collect(1.0:length(pᵢ_coeffs))                 # Anti-derivative of pᵢ
        qᵢ_func(x) = sum(qᵢⱼ*x^Float64(j) for (j, qᵢⱼ) in enumerate(qᵢ_coeffs))
        push!(q_func, qᵢ_func)
    end
    # Use anti-derivatives to evaluate integration matrix. These are integrals
    # between nodes in our subinterval for each polynomial antiderivate 'basis'
    # vector.
    return [qᵢ_func(t_part[i + 1]) - qᵢ_func(t_part[i]) for i in 1:(N + 1), qᵢ_func in q_func]
end
function integration_matrix_legendre(N::Int)
    gl_quad = gausslegendre(N)
    integration_matrix_legendre(N, vcat(-1, gl_quad[1], 1))
end


"""
Algorithm from https://doi.org/10.1137/09075740X
Note passing integration matrix S as argument is much more efficient for testing
as the same matrix may otherwise be calculated multiple times for different
function calls.
"""
function IDC_FE(S, f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, M)
    η = zeros(N + 1)
    η[1] = α

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        for _ in 2:p
            η_old = copy(η)
            for m in 1:M
                k = j*M + m
                I = (j*M + 1):((j + 1)*M+ 1)
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
        end
    end

    return (t, η)
end
function IDC_FE(f, a, b, α, N, p)
    return IDC_FE(integration_matrix_equispaced(p - 1), f, a, b, α, N, p)
end

"""
Same algorithm as 'IDC_FE' but storing all prediction and correction
levels in a matrix output. This is useful for calculating stability regions at
different correction stages.
"""
function IDC_FE_correction_levels(S, f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, M)
    η = zeros(Complex, N + 1, p)
    η[1, :] .= α

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

function IDC_RK2(S, f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, M)
    η = zeros(N + 1)
    η[1] = α

    for j in 0:(J - 1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1] = RK_time_step_explicit(f, Δt, t[k], η[k]; RK_method = RK2_midpoint)
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
function IDC_RK2(f, a, b, α, N, p)
    return IDC_RK2(integration_matrix_equispaced(p - 1), f, a, b, α, N, p)
end

function IDC_RK_general(S, f, a, b, α::T, N, p; RK_method::RKMethod = RK2_midpoint) where T
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, M)
    η = zeros(T, N + 1)
    η[1] = α

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
"""
function IDC_FE_single(S, f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    η = zeros(N + 1)
    η[1] = α

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
function IDC_FE_single(f, a, b, α, N, p)
    return IDC_FE_single(integration_matrix_equispaced(N), f, a, b, α, N, p)
end

"""
Same algorithm as 'IDC_FE_single' but storing all prediction and correction
levels. This is useful for calculating stability regions at different correction
stages.
"""
function IDC_FE_single_correction_levels(S, f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    η = zeros(Complex, N + 1, p)
    η[1, :] .= α

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
"""
function SDC_FE(S, f, a, b, α, N, p)
    # Initialise variables
    M = p + 1
    J = fld(N, M)
    η = zeros(N + 1)
    η[1] = α
    # Sandwich legendre nodes within subintervals
    legendre_quadrature = gausslegendre(p)
    int_scale = (b - a)/J
    legendre_t = (legendre_quadrature[1] .+ 1).*int_scale/2
    t = reduce(vcat, [vcat(j*int_scale, j*int_scale .+ legendre_t) for j in 0:(J - 1)])
    t = vcat(t, b)
    Δt = t[2:end] .- t[1:end - 1]    # No longer constant

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
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))*int_scale/2
                η[k + 1] = η[k] + Δt[k]*(f(t[k], η[k]) - f(t[k], η_old[k])) + ∫fₖ
            end
        end
    end

    return (t, η)
end
"""
In order for our quadrature to have sufficient order by the last correction
step, we need to use the same number of nodes as the order we wish to acheive
with this step.
It is not possible to make a reduced stencil algorithm for this as our legendre
nodes are fixed within their subintervals.
"""
function SDC_FE(f, a, b, α, N, p)
    return SDC_FE(integration_matrix_legendre(p), f, a, b, α, N, p)
end

function SDC_FE_single(S, f, a, b, α, N, p)
    # Initialise variables
    η = zeros(N + 1)
    η[1] = α
    gl = gausslegendre(N - 1)
    t_legendre = (gl[1] .+ 1)/2             # legendre nodes in [0, 1]
    t_legendre = t_legendre.*(b - a) .+ a   # Legendre nodes in [a, b]
    t = vcat(a, t_legendre, b)
    Δt = t[2:end] .- t[1:end - 1]           # No longer constant

    # Prediction loop
    for k in 1:N
        η[k + 1] = η[k] + Δt[k]*f(t[k], η[k])
    end
    # Correction loop
    I = 2:N     # Use non-endpoint nodes for quadrature
    for _ in 2:p
        η_old = copy(η)
        for k in 1:N
            ∫fₖ = dot(S[k, :], f.(t[I], η_old[I]))*(b - a)/2
            η[k + 1] = η[k] + Δt[k]*(f(t[k], η[k]) - f(t[k], η_old[k])) + ∫fₖ
        end
    end

    return (t, η)
end
function SDC_FE_single(f, a, b, α, N, p)
    return SDC_FE(integration_matrix_legendre(N - 1), f, a, b, α, N, p)
end

"""
The sequential form of our RIDC algorithm is relatively simple: we've simply
added more space in each subinterval whilst interpolating over the same number
of nodes. The main drawback to this potential parallelisability is that only the
less stable uniform nodes may be used.
"""
function RIDC_FE_sequential(S, f, a, b, α, N, K, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, K)
    η = zeros(Complex, N + 1)
    η[1] = α

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
function RIDC_FE_sequential(f, a, b, α, N, K, p)
    return RIDC_FE_sequential(integration_matrix_equispaced(p - 1), f, a, b, α, N, K, p)
end

function RIDC_FE_sequential_correction_levels(S, f, a, b, α, N, K, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, K)
    η = zeros(Complex, N + 1, p)
    η[1, :] .= α

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
"""
function RIDC_FE_sequential_reduced_stencil(S::Array{Matrix{Float64}}, f, a, b, α, N, K, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, K)
    η = zeros(N + 1)
    η[1] = α

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
function RIDC_FE_sequential_reduced_stencil(f, a, b, α, N, K, p)
    S = [integration_matrix_equispaced(l) for l in 1:(p - 1)]
    return RIDC_FE_sequential_reduced_stencil(S, f, a, b, α, N, K, p)
end
end