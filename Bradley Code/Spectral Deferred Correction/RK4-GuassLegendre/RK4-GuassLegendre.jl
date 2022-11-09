# Import packages
using Plots, GLM, DataFrames

# Function to get the normalised legendre polynomial of order k, P_1(x)=1, P_2(x)=x
function legendre_pol(k)
    pol1 = Float64[1]
    pol2 = Float64[0, 1]
    pols = [pol1, pol2]
    for i in 3:k+1
        newpol = [0; pols[i-1]] * (2*(i-1)-1)/(i-1) - [pols[i-2]; 0; 0] * ((i-1)-1)/(i-1)
        push!(pols, newpol)
    end
    return pols[k+1]
end

# Function to find all the k unique roots of the order k legendre polynomial via Newton-Raphson with s iterations
function legendre_roots(k, s, digits)
    coeffs = legendre_pol(k)
    # Legendre polynomial of order k
    pol(x) = sum(coeffs .* x .^ collect(0:1:k))
    # Derivative of polynomial
    dpol(x) = sum(coeffs[2:k+1] .* collect(1:1:k) .* x .^ collect(0:1:k-1))
    # Discretise [0, 1]
    xx = 1/k:1/k:1
    # Use Newton-Raphson with s steps at each of these points
    roots = Float64[]
    for x_init in xx
        x = x_init
        for step in 1:s
            x = x - pol(x)/dpol(x)
        end
        push!(roots, x)
    end
    roots = round.(roots, digits = digits)
    roots = unique(roots)
    roots = union(roots, -roots[roots .!= 0])
    nroots = length(roots)
    if nroots != k
        @error "Expected $k roots, calculated $nroots"
    end
    return roots
end

# Function to integrate area of function f between a and b using Guass-Legendre quadrature
function guass_legendre_int(f, k, a, b)
    # Coeffs of the k'th polynomial
    coeffs = legendre_pol(k)
    # Derivative of the k'th polynomial
    dpol(x) = sum(coeffs[2:k+1] .* collect(1:1:k) .* x .^ collect(0:1:k-1))
    # Want to integrate between -1 to 1 so perform transformation
    g(u) = f(a + 0.5*(b-a)*(u+1))

    # Here I have set s=100, rounded to digits=10, may be inadequate.
    roots = legendre_roots(k, 100, 10)
    weights = 1 ./ ((1 .- roots .^ 2) .* dpol.(roots) .^ 2)

    int = (b-a)*sum(weights .* g.(roots))
    return int
end

# f is f(u, t), a is lower bound, b is upper bound, u_init is initial value u₀
# k is number of nodes used in Euler method
# n is number of corrections, U is true u(t), F is true u'(t)
function spectral_correction(f, a, b, u_init, k, n, U, F, disp)
    # Fine discretisation for plotting known functions and interpolations 
    tt = a:(b-a)/1000:b

    # Inital u(t) approximate points
    h = (b-a)/k
    t = [i*h + a for i in 0:k]
    u_hat = zeros(k+1)
    u_hat[1] = u_init
    for i in 1:k
        k1 = f(u_hat[i], t[i])
        k2 = f(u_hat[i]+0.5*h*k1, t[i]+0.5*h)
        k3 = f(u_hat[i]+0.5*h*k2, t[i]+0.5*h)
        k4 = f(u_hat[i]+h*k3, t[i]+h)
        u_hat[i+1] = u_hat[i] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    end

    u_mat = zeros(k+1, n+1)
    u_mat[:, 1] = u_hat
    # Performs the n corrections 
    for i in 1:n
        f_hat = f.(u_hat, t)

        # Calculate polynomial interpolation f_pol(t) of f(u, t) using equidistant nodes
        function f_pol(x)
            S = 0
            for j in 1:k+1
                u = union(1:j-1, j+1:k+1)
                S = S + f_hat[j]*prod((x .- t[u]) ./ (t[j] .- t[u]))
            end
            return(S)
        end

        # Residuals
        # Guass-Legendre Quadrature method using 10 nodes
        int_hat = [guass_legendre_int(f_pol, 10, a, b) for b in t]
        res = int_hat .+ u_init .- u_hat
        
        # Approximate errors
        errs = zeros(k+1)
        errs[1] = 0
        for j in 1:k
            errs[j+1] = errs[j] + h*(f(u_hat[j] + errs[j], t[j]) - f(u_hat[j], t[j])) + res[j+1] - res[j]
        end

        # Correct the approximations for u(t) using approximated errors
        u_hat_new = u_hat + errs
        u_mat[:, i+1] = u_hat_new

        if disp && U != nothing && F != nothing
            p1 = plot(tt, F.(tt), labels = "True f(u, t)")
            scatter!(t, f_hat, labels = "Approx f(u, t)", markersize = 1.5, markerstrokewidth = 0)
            plot!(tt, f_pol.(tt), labels = "Interpolated f(u, t)")
            p2 = plot(t, res, labels = "Approx residuals")
            p3 = plot(t, U.(t) - u_hat, labels = "True errors")
            plot!(t, errs, labels = "Approx errors")
            p4 = plot(tt, U.(tt), labels = "True u(t)")
            scatter!(t, u_hat, labels = "Approx u(t)", markersize = 1.5, markerstrokewidth = 0)
            scatter!(t, u_hat_new, labels = "Correction u(t)", markersize = 1.5, markerstrokewidth = 0)
            display(plot(p1, p2, p3, p4, plot_title = "Correction $i", legend = :outertop))
        end
        u_hat = u_hat_new
    end

    if disp && U != nothing
        p = plot(tt, U.(tt), labels = "True u(t)")
        scatter!(t, u_mat[:, 1], labels = "Initial Approx", legend = :outertopright, markersize = 1.5, markerstrokewidth = 0)
        for i in 1:n
            scatter!(t, u_mat[:, i+1], labels = "Correction $i", legend = :outertopright, markersize = 1.5, markerstrokewidth = 0)
        end
        display(p)
    end
    return(abs(U.(t[end]) - u_mat[end, end]))
end

function main()
    # ODE
    # u'(t) = f(u(t), t), t ∈ [a, b]
    function f(u, t)
        if t == 0
            return 5
        elseif t == 1
            return 0
        else
            return (u/t) - 5*(2*u/(1-t) + u)
        end
    end


    # Initial conditions
    # t₀ = a = 0, u₀ = u(t₀) = 0

    # Initialise parameters
    a = 0
    b = 1
    u_init = 0
    # Number of time steps (k+1 nodes)
    k = 20
    # Number of corrections
    n = 3

    # True solution u(t)
    U(t) = 5*exp(-5*t)*t*(1-t)^10
    # True derivative f(t) = u'(t)
    F(t) = (U(t)/t) - 5*(2*U(t)/(1-t) + U(t))

    final_err = spectral_correction(f, a, b, u_init, k, n, U, F, true)
    println(final_err)

    I = 5:20
    niterations = length(I)
    log_final_errs = Vector{Float64}(undef, niterations)
    log_hs = Vector{Float64}(undef, niterations)

    # Calculate log global error for different numbers of nodes i+1 
    for i in I
        log_final_err = log(spectral_correction(f, a, b, u_init, i, n, U, F, false))
        log_final_errs[i+1-minimum(I)] = log_final_err
        log_h = log((b-a)/i)
        log_hs[i+1-minimum(I)] = log_h
    end

    data = DataFrame(X = log_hs, Y = log_final_errs)
    fm = @formula(Y ~ X)
    lfit = lm(fm, data)
    coefs = coef(lfit)

    x_min = minimum(log_hs)
    x_max = maximum(log_hs)
    xx = x_min:(x_max-x_min)/100:x_max
    y_min = minimum(log_final_errs)
    y_max = maximum(log_final_errs)
    yy1 = (xx .- x_min) .+ y_min .+ 1
    yy = 4*(n+1)*(xx .- x_min) .+ y_min .+ 1
    plot(xx, yy1, title = "Convergence plot with $n corrections", labels = "Order 1 test line", linestyle = :dash, legend = :outertopright, ylims = [y_min, y_max])
    plot!(xx, yy, labels = "Order $(4*(n+1)) test line", linestyle = :dash)
    plot!(log_hs, log_final_errs, labels = "Global error", xlabel = "Log step size", ylabel = "Global log error")
    display(Plots.abline!(coefs[2], coefs[1], labels = "Global error regression", xlabel = "Log step size", ylabel = "Global log error"))
end

main()
