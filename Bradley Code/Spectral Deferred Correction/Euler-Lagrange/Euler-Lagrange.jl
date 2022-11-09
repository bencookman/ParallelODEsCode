# Import packages
using Plots, GLM, DataFrames

# Function to calculate closed Newton-Cotes Quadrature weights
function newton_cotes_weights(t, n)
    weights = zeros(n+1)
    for j in 1:n+1
        u = union(1:j-1, j+1:n+1)
        coeff = [1, -t[u[1]]] / (t[j] - t[u[1]])
        for l in 2:n
            coeff = ([coeff; 0] - t[u[l]]*[0; coeff]) / (t[j] - t[u[l]])
        end
        evalb = sum((coeff ./ collect(n+1:-1:1)) .* t[end] .^ collect(n+1:-1:1))
        evala = sum((coeff ./ collect(n+1:-1:1)) .* t[1] .^ collect(n+1:-1:1))
        weights[j] = evalb - evala
    end
    return(weights)
end

# Function to approximately integrate a polynomial interpolation of f(u, t) using Newton-Cotes
function newton_cotes_integration(t, n, f_pol)
    int_hat = zeros(length(t))
    for j in 2:length(t)
        sub_t = [i*(t[j]-t[1])/n + t[1] for i in 0:n]
        int_hat[j] = sum(f_pol.(sub_t) .* newton_cotes_weights(sub_t, n))
    end
    return(int_hat)
end

# f is f(u, t), a is lower bound, b is upper bound, u_init is initial value u₀
# k+1 is number of Euler nodes and number of quadrature Coefficients
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
        u_hat[i+1] = u_hat[i] + h*f(u_hat[i], t[i])
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
        # Newton-Cotes Quadrature method
        int_hat = newton_cotes_integration(t, k, f_pol)
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
    f(u, t) = -2*t*u

    # Initial conditions
    # t₀ = a = 0, u₀ = u(t₀) = 0

    # Initialise parameters
    a = 0
    b = 2
    u_init = 1
    # Number of time steps (k+1 nodes)
    k = 10
    # Number of corrections
    n = 10

    # True solution u(t)
    U(t) = exp(-t^2)
    # True derivative f(t) = u'(t)
    F(t) = -2*t*exp(-t^2)

    final_err = spectral_correction(f, a, b, u_init, k, n, U, F, true)
    println(final_err)

    I = 1:15
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
    yy = (n+1)*(xx .- x_min) .+ y_min .+ 1
    plot(xx, yy1, title = "Convergence plot with $n corrections", labels = "Order 1 test line", linestyle = :dash, legend = :outertopright, ylims = [y_min, y_max])
    plot!(xx, yy, labels = "Order $(n+1) test line", linestyle = :dash)
    plot!(log_hs, log_final_errs, labels = "Global error", xlabel = "Log step size", ylabel = "Global log error")
    display(Plots.abline!(coefs[2], coefs[1], labels = "Global error regression", xlabel = "Log step size", ylabel = "Global log error"))
end

main()
