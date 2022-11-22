module ProjectTools

using LinearAlgebra, Statistics, PyCall, FastGaussQuadrature

export
    err_abs,
    err_rel,
    err_cum,
    err_norm,
    err_max,

    integration_matrix,
    integration_matrix_equispaced,
    integration_matrix_legendre_inner,

    IDC_FE,
    IDC_RK2,
    IDC_FE_single,
    SDC_FE,
    SDC_FE_single,
    RIDC_FE_sequential,
    RIDC_FE_sequential_reduced_stencil

err_abs(exact, approx) = abs.(exact - approx)
err_rel(exact, approx) = err_abs(exact, approx) ./ exact
err_cum(exact, approx) = [sum(err_abs(exact[1:i], approx[1:i])) for i in 1:length(data_correct)]
err_norm(exact, approx, p) = norm(err_abs(exact, approx), p)
err_max(exact, approx)  = maximum(err_abs(exact, approx))

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

function integration_matrix_legendre_inner(N::Int)
    scipy_interpolate = pyimport("scipy.interpolate")

    # Calculate anti-derivatives of polynomial interpolants
    q_func = []
    gl_quad = gausslegendre(N)
    t = gl_quad[1]              # Legendre nodes in [-1, 1]
    for i in 1:N
        yᵢ = zeros(Float64, N)
        yᵢ[i] = 1.0
        pᵢ_coeffs = scipy_interpolate.lagrange(t, yᵢ).c |> reverse              # Lagrange interpolant to yᵢ at x
        qᵢ_coeffs = pᵢ_coeffs ./ collect(1.0:length(pᵢ_coeffs))                 # Anti-derivative of pᵢ
        qᵢ_func(x) = sum(qᵢⱼ*x^Float64(j) for (j, qᵢⱼ) in enumerate(qᵢ_coeffs))
        push!(q_func, qᵢ_func)
    end
    # Use anti-derivatives to evaluate integration matrix. These are integrals
    # between nodes in our subinterval for each polynomial antiderivate 'basis'
    # vector.
    t_closed = vcat(-1, t, 1)
    return [qᵢ_func(t_closed[i + 1]) - qᵢ_func(t_closed[i]) for i in 1:(N + 1), qᵢ_func in q_func]
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

    for j in 0:(J-1)
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
                I = (j*M + 1):(j*(M + 1) + 1)
                ∫fₖ = dot(S[m, I], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
        end
    end

    return η
end
function IDC_FE(f, a, b, α, N, p)
    return IDC_FE(integration_matrix_equispaced(p - 1), f, a, b, α, N, p)
end

function IDC_RK2(S, f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N + 1) |> collect
    Δt = (b - a)/N
    M = p - 1
    J = fld(N, M)
    η = zeros(N + 1)
    η[1] = α

    for j in 0:(J-1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1] = η[k] + 0.5Δt*(f(t[k], η[k]) + f(t[k + 1], η[k] + Δt*f(t[k], η[k])))
        end
        # Correction loop
        for _ in 2:fld(p, 2)
            η_old = copy(η)
            for m in 1:M
                k = j*M + m
                ∫fₖ = sum(S[m, i]*f(t[j*M + i], η_old[j*M + i]) for i in 1:(M + 1))
                K₁ = f(t[k], η[k]) - f(t[k], η_old[k])
                K₂ = f(t[k + 1], η[k] + Δt*(K₁ + ∫fₖ)) - f(t[k + 1], η_old[k + 1])
                η[k + 1] = η[k] + Δt*(0.5K₁ + 0.5K₂ + ∫fₖ)
            end
        end
    end

    return η
end
function IDC_RK2(f, a, b, α, N, p)
    return IDC_RK2(integration_matrix_equispaced(p - 1), f, a, b, α, N, p)
end

"""
Integral deferred correction with a single subinterval (uses static 
interpolatory quadrature over all N + 1 nodes). To improve this could use a
reduced size integration matrix of minimum necessary size (p - 1 = M) with
rolling usage of interpolation nodes.
"""
function IDC_FE_single(S, f, a, b, α, N, p)
    # Initialise variables
    t = range(a, b, N+1) |> collect
    Δt = (b - a)/N
    η = zeros(N+1)
    η[1] = α

    # Prediction loop
    for m in 1:N
        η[m + 1] = η[m] + Δt*f(t[m], η[m])
    end
    # Correction loop
    for _ in 2:p
        η_old = copy(η)
        for m in 1:N
            η[m + 1] = η[m] + Δt*(f(t[m], η[m]) - f(t[m], η_old[m])) + Δt*sum(S[m, i]*f(t[i], η_old[i]) for i in 1:(N + 1))
        end
    end

    return η
end
function IDC_FE_single(f, a, b, α, N, p)
    return IDC_FE_single(integration_matrix_equispaced(N), f, a, b, α, N, p)
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

    for j in 0:(J-1)
        # Prediction loop
        for m in 1:M
            k = j*M + m
            η[k + 1] = η[k] + Δt[k]*f(t[k], η[k])
        end
        # Correction loop
        I = (j*M + 2):((j + 1)*M)
        for _ in 2:p
            η_old = copy(η)
            for m in 1:M
                k = j*M + m
                # Only those in the middle of the subinterval are legendre nodes
                ∫fₖ = dot(S[m, :], f.(t[I], η_old[I]))*int_scale/2
                η[k + 1] = η[k] + Δt[k]*(f(t[k], η[k]) - f(t[k], η_old[k])) + ∫fₖ
            end
        end
    end

    return η
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
    I = 2:N
    for _ in 2:p
        η_old = copy(η)
        for k in 1:N
            # Use non-endpoint nodes for quadrature
            ∫fₖ = dot(S[k, :], f.(t[I], η_old[I]))*(b - a)/2
            η[k + 1] = η[k] + Δt[k]*(f(t[k], η[k]) - f(t[k], η_old[k])) + ∫fₖ
        end
    end

    return η
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
    η = zeros(N + 1)
    η[1] = α

    for j in 0:(J-1)
        # Prediction loop
        for m in 1:M
            k = j*K + m
            η[k + 1] = η[k] + Δt*f(t[k], η[k])
        end
        # Correction loop
        for _ in 2:p
            η_old = copy(η)
            for m in 1:M
                k = j*K + m
                I = (j*K + 1):(j*K + M + 1)
                ∫fₖ = dot(S[m, I], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
            for m in (M + 1):K
                k = j*K + m
                I = (j*K + m - M):(j*K + m)
                ∫fₖ = dot(S[M, I], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
        end
    end

    return η
end
function RIDC_FE_sequential(f, a, b, α, N, K, p)
    return RIDC_FE_sequential(integration_matrix_equispaced(p - 1), f, a, b, α, N, K, p)
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
                ∫fₖ = dot(S[l][m, I], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
            for m in (M + 1):K
                k = j*K + m
                I = (j*K + m - M):(j*K + m)
                ∫fₖ = dot(S[l][M, I], f.(t[I], η_old[I]))
                η[k + 1] = η[k] + Δt*(f(t[k], η[k]) - f(t[k], η_old[k])) + Δt*∫fₖ
            end
        end
    end

    return η
end
function RIDC_FE_sequential_reduced_stencil(f, a, b, α, N, K, p)
    S = [integration_matrix_equispaced(l) for l in 1:(p - 1)]
    return RIDC_FE_sequential_reduced_stencil(S, f, a, b, α, N, K, p)
end

end