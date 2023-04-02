using
    Dates,
    Plots,
    Statistics,
    LinearAlgebra,
    LaTeXStrings,
    FastGaussQuadrature,
    PyCall,
    Measures

include("ProjectTools.jl")

using .ProjectTools

const MATPLOTLIB = pyimport("matplotlib")
const RCPARAMS = PyDict(MATPLOTLIB["rcParams"])
RCPARAMS["mathtext.fontset"] = "cm"
RCPARAMS["xtick.major.pad"] = 10


"""
For cases where we are using multiple large weights matrices where we have a
large number of quadrature nodes, it is beneficial to have them precalculated. A
better solution would be to calculate these once and store them in an external
file (probably .csv for conveneience) somewhere, to be fetched as needed.
"""
const INTEGRATION_MATRIX_ARRAY_0_TO_N = []
const INTEGRATION_MATRIX_ARRAY_UNIFORM = []
const INTEGRATION_MATRIX_ARRAY_CHEBYSHEV = []

"""
Giving only maximum size as argument is reasonable as the matrices of smaller
size are significantly easier to generate, so they are of little cost. This also
means the above 'INTEGRATION_MATRIX_ARRAY_...'s are automatically ordered.
"""
function fill_integration_matrix_array_0_to_N(max_size)
    current_size = size(INTEGRATION_MATRIX_ARRAY_0_TO_N, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    new_N_array = (current_size + 1):max_size
    push!(INTEGRATION_MATRIX_ARRAY_0_TO_N, [integration_matrix_uniform(N + 1) for N in new_N_array]...)
    nothing
end

function fill_integration_matrix_array_uniform(max_size)
    current_size = size(INTEGRATION_MATRIX_ARRAY_UNIFORM, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    new_N_array = (current_size + 1):max_size
    t_nodes_array = [range(-1, 1, N + 1) for N in new_N_array]
    push!(INTEGRATION_MATRIX_ARRAY_UNIFORM, [integration_matrix(t_nodes, t_nodes) for t_nodes in t_nodes_array]...)
    nothing
end

function fill_integration_matrix_array_legendre(max_size)
    current_size = size(INTEGRATION_MATRIX_ARRAY_LEGENDRE, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    new_N_array = (current_size + 1):max_size
    push!(INTEGRATION_MATRIX_ARRAY_LEGENDRE, [integration_matrix_legendre(N) for N in new_N_array]...)
    nothing
end

function fill_integration_matrix_array_lobatto(max_size)
    current_size = size(INTEGRATION_MATRIX_ARRAY_LOBATTO, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    new_N_array = (current_size + 1):max_size
    push!(INTEGRATION_MATRIX_ARRAY_LOBATTO, [integration_matrix_lobatto(N + 1) for N in new_N_array]...)
    nothing
end

function fill_integration_matrix_array_chebyshev(max_size)
    current_size = size(INTEGRATION_MATRIX_ARRAY_CHEBYSHEV, 1)
    (current_size >= max_size) && throw(error("now new matrices needed"))
    new_N_array = (current_size + 1):max_size
    t_nodes_array = [(gausschebyshev(N)[1], [-1, gausschebyshev(N)[1]..., 1]) for N in new_N_array]
    push!(INTEGRATION_MATRIX_ARRAY_CHEBYSHEV, [
        integration_matrix(quadrature_nodes_array, time_step_nodes_array)
        for (quadrature_nodes_array, time_step_nodes_array) in t_nodes_array
    ]...)
    nothing
end


function integration_matrix_uniform_test()
    # Set up test
    # t_end = 1.0
    # f(t) = cos(t)
    # integral_exact = sin(t_end)
    # t_end = 1
    # f(t) = cos(t)^2
    # integral_exact = t_end/2 + sin(2*t_end)/4
    t_end = 1
    f(t) = sqrt(t)
    integral_exact = 2*(t_end)^(1.5)/3
    # t_end = 0.001
    # f(t) = cos(t)*exp(sin(t))
    # integral_exact = exp(sin(t_end)) - 1

    # Do test
    integral_approximations = Array{Float64, 1}(undef, 0)
    sum_resolutions = 2:2:30
    for sum_resolution in sum_resolutions
        t_sample = 0:(sum_resolution + 1) |> collect
        f_sample = f.(t_sample)
        S = integration_matrix_uniform(sum_resolution)
        integral_approx = sum(S[1, i] .* f_sample[i] for i in 1:(sum_resolution+1))

        println(integral_approx)
        push!(integral_approximations, integral_approx)
    end

    # Plot test results
    integral_error = abs.(integral_exact .- integral_approximations)
    Δt_values = t_end./sum_resolutions
    test_plot = plot(
        Δt_values, integral_error,
        xscale=:log10, yscale=:log10, xlabel=L"\Delta t", ylabel=L"||E||",
        size=(1200, 900), thickness_scaling=1.5
    )
    dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    fname = "Ben Code/output/$dtstring-int-matrix-test.png"
    savefig(test_plot, fname)
end

""" similar test to above """
function integration_matrix_uniform_test_0_to_N()
    # Set up variables
    N_array = 1:20
    # g(t) = cos(t)                     # Integrand
    # G(t) = sin(t)                     # Primative of g
    g(t) = cos(t)^2
    G(t) = sin(2t)/4 + t/2
    # g(t) = log(1 + t)
    # G(t) = (1 + t)*log(1 + t) - t
    err_array = []

    # Run integral test
    I = G(1) - G(0)
    # println(I)
    for N in N_array
        t = 0:N
        g̲ = g.(t)
        W = INTEGRATION_MATRIX_ARRAY_0_TO_N[N]
        I_N = dot(W[1, :], g̲)
        println(I_N)

        err = abs(I - I_N)
        push!(err_array, err)
    end

    # Plot integral test results
    test_plot = plot(
        N_array, err_array,
        xscale = :log10, yscale = :log10, xlabel = "N", ylabel = "|E|",
        thickness_scaling = 1.0
    )
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/$dtstring-int-matrix-test.png"
    # savefig(test_plot, fname)
    display(test_plot)
end

"""
Tests of matrices using different node schemes using the classic Runge
phenomenon example where g(t) = 1/(1 + 25t^2). This is a better analogy for an
IDC algorithm with fixed time domain [a, b] = [-1, 1].

Could turn this into two functions with reduced boilerplate code: one for closed
quadratures (uniform and Lobatto), and one for open quadratures (Legendre and
Chebyshev)
"""

"""
Uniform nodes result in the Runge phenomenon rather quickly, so the approximate
integral from t_1 to t_2 diverges fast away from 0.
"""
function integration_matrix_test_classic_uniform(N_array)
    g(t) = 1/(1 + 25t^2)
    G(t) = 0.2atan(5t)

    # Run integral test
    err_array = []
    for N in N_array
        t = range(-1, 1, N + 1)
        g̲ = g.(t)
        W = INTEGRATION_MATRIX_ARRAY_UNIFORM[N]
        I_N = dot(W[1, :], g̲)
        I = G(t[2]) - G(t[1])     # Exact integral between consecutive time points changes

        # err = abs(I - I_N)/abs(I)
        err = abs(I_N)              # Absolute value of I_N blows up quickly when it should not
        push!(err_array, err)
    end

    # Plot integral test results
    test_plot = plot(
        N_array, err_array,
        yscale = :log10, xlabel = L"N", ylabel = L"|I_N|",
        thickness_scaling = 1.0,
        legend = false,
        size = (900, 800),
        markershape = :circle, color = :blue,
        topmargin = 2mm
    )
    display(test_plot)

    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/$dtstring-int-matrix-test_classic.png"
    # savefig(test_plot, fname)
end

function integration_matrix_test_classic_legendre(N_array)
    g(t) = 1/(1 + 25t^2)
    G(t) = 0.2atan(5t)

    # Run integral test
    err_array = []
    for N in N_array
        t_legendre = gausslegendre(N)[1]
        t = [-1, t_legendre..., 1]
        g̲ = g.(t[2:end - 1])
        W = INTEGRATION_MATRIX_ARRAY_LEGENDRE[N]
        I_N = dot(W[1, :], g̲)
        I = G(t[2]) - G(t[1])       # Exact integral between consecutive time points changes

        err = abs(I - I_N)/abs(I)   # Relative error of I_N against I
        # err = abs(I_N)
        push!(err_array, err)
    end

    # Plot integral test results
    test_plot = plot(
        N_array, err_array,
        yscale = :log10, xlabel = L"N", ylabel = L"|E|",
        thickness_scaling = 1.0,
        legend = false,
        size = (900, 800),
        markershape = :circle, color = :blue,
        topmargin = 2mm
    )
    display(test_plot)

    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/$dtstring-int-matrix-test_classic.png"
    # savefig(test_plot, fname)
end

function integration_matrix_test_classic_lobatto(N_array)
    g(t) = 1/(1 + 25t^2)
    G(t) = 0.2atan(5t)

    # Run integral test
    err_array = []
    for N in N_array
        t = gausslobatto(N + 1)[1]
        g̲ = g.(t)
        W = INTEGRATION_MATRIX_ARRAY_LOBATTO[N]
        I_N = dot(W[1, :], g̲)
        I = G(t[2]) - G(t[1])       # Exact integral between consecutive time points changes

        err = abs(I - I_N)/abs(I)   # Relative error of I_N against I
        # err = abs(I_N)
        push!(err_array, err)
    end

    # Plot integral test results
    test_plot = plot(
        N_array, err_array,
        yscale = :log10, xlabel = L"N", ylabel = L"|E|",
        thickness_scaling = 1.0,
        legend = false,
        size = (900, 800),
        markershape = :circle, color = :blue,
        topmargin = 2mm
    )
    display(test_plot)

    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/$dtstring-int-matrix-test_classic.png"
    # savefig(test_plot, fname)
end

function integration_matrix_test_classic_chebyshev(N_array)
    g(t) = 1/(1 + 25t^2)
    G(t) = 0.2atan(5t)

    # Run integral test
    err_array = []
    for N in N_array
        t_chebyshev = gausschebyshev(N)[1]
        t = [-1, t_chebyshev..., 1]
        g̲ = g.(t[2:end - 1])
        W = INTEGRATION_MATRIX_ARRAY_CHEBYSHEV[N]
        I_N = dot(W[1, :], g̲)
        I = G(t[2]) - G(t[1])     # Exact integral between consecutive time points changes

        err = abs(I - I_N)/abs(I)   # Relative error of I_N against I
        # err = abs(I_N)
        push!(err_array, err)
    end

    # Plot integral test results
    test_plot = plot(
        N_array, err_array,
        yscale = :log10, xlabel = L"N", ylabel = L"|E|",
        thickness_scaling = 1.0,
        legend = false,
        size = (900, 800),
        markershape = :circle, color = :blue,
        topmargin = 2mm
    )
    display(test_plot)

    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/$dtstring-int-matrix-test_classic.png"
    # savefig(test_plot, fname)
end


