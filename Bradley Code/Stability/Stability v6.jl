using Plots

# Function to get Lagrange weights for Newton-Cotes integration
function lagrange_weights(a, b, n, k)
    h = (b-a)/(n*k)
    nodes = [a+h*(i-1) for i in 1:k+1]

    k = length(nodes)
    weights = zeros(k-1, k)
    for m in 1:k-1
        for i in 1:k
            f_pol = [1]
            for j in union(1:i-1, i+1:k)
                f_pol = [f_pol; 0]*-nodes[j]/(nodes[i]-nodes[j]) + [0; f_pol]/(nodes[i]-nodes[j])
            end
            weights[m, i] = sum((1 ./ collect(1:k)) .* f_pol .* nodes[m+1] .^ collect(1:k)) - sum((1 ./ collect(1:k)) .* f_pol .* nodes[m] .^ collect(1:k))
        end
    end
    return weights
end

# RIDC Euler-Euler Method

function RIDC_Euler(f, a, b, u_init, W, n, k, l)
    big_h = (b-a)/n
    h = big_h/k

    u_full_matrix = zeros(Number, l+1, n*k+1)
    for p in 1:n
        nodes = [a+big_h*(p-1)+h*(i-1) for i in 1:k+1]

        # Initial prediction via Euler Method
        u_hat = zeros(Number, k+1)
        u_hat[1] = u_init
        for i in 1:k
            u_hat[i+1] = u_hat[i] + h*f(u_hat[i], nodes[i])
        end

        # Initialise correction matrix with its initial conditions
        u_matrix = zeros(Number, l+1, k+1)
        u_matrix[1, :] = u_hat
        u_matrix[:, 1] .= u_init

        # Corrections section via integral deferred correction
        for l in 1:l
            f_hat = f.(u_matrix[l, :], nodes)
            f_int = W * f_hat
            errors = zeros(Number, k+1)
            for i in 1:k
                G = f(u_matrix[l+1, i], nodes[i]) - f(u_matrix[l, i], nodes[i])

                # Find approximate errors for current correction level
                errors[i+1] = errors[i] + h*G - (u_matrix[l, i+1] - u_matrix[l, i]) + f_int[i]

                # Update u_hat value to next correction level using error
                u_matrix[l+1, i+1] = u_matrix[l, i+1] + errors[i+1]
            end
        end
        u_init = u_matrix[end, end]
        u_full_matrix[:, 1+k*(p-1):1+k*p] = u_matrix
    end
    return u_full_matrix
end

# RIDC RK2-RK2 Method

function RIDC_RK2(f, a, b, u_init, W, n, k, l)
    big_h = (b-a)/n
    h = big_h/k

    nodes = [a+h*(i-1) for i in 1:k+1]

    u_full_matrix = zeros(Number, l+1, n*k+1)
    for p in 1:n
        nodes = [a+big_h*(p-1)+h*(i-1) for i in 1:k+1]

        # Initial prediction via Runge-Kutta 2 method (Heun's Method)
        u_hat = zeros(Number, k+1)
        u_hat[1] = u_init
        for i in 1:k
            u_hat[i+1] = u_hat[i] + h*f(u_hat[i] + h*f(u_hat[i], nodes[i])/2, nodes[i] + h/2)
        end

        # Initialise correction matrix with its initial conditions
        u_matrix = zeros(Number, l+1, k+1)
        u_matrix[1, :] = u_hat
        u_matrix[:, 1] .= u_init

        # Corrections section via integral deferred correction
        for l in 1:l
            f_hat = f.(u_matrix[l, :], nodes)
            f_int = W * f_hat
            for i in 1:k
                # Update u_hat value to next correction level
                K1 = h*(f(u_matrix[l+1, i], nodes[i]) - f(u_matrix[l, i], nodes[i]))
                K2 = h*(f(u_matrix[l+1, i] + K1 + f_int[i], nodes[i+1]) - f(u_matrix[l, i+1], nodes[i+1]))
                u_matrix[l+1, i+1] = u_matrix[l+1, i] + K1/2 + K2/2 + f_int[i]
            end
        end
        u_init = u_matrix[end, end]
        u_full_matrix[:, 1+k*(p-1):1+k*p] = u_matrix
    end
    return u_full_matrix
end

# A-Stability Test

# ODE u'(t) = f(u, t) = λ*u(t), λ ∈ ℂ

# Initial Condition u(t = 0) = 1

# Solution u(t) = exp(k*t)

function Stability_Test(; method::String = "Euler", xbounds::Vector = [-2.5, 2.5], ybounds::Vector = [-2, 2], res::Integer = 1000)
    xrange = range(xbounds..., length = res)
    yrange = range(ybounds..., length = res)

    conv = []
    grid = Iterators.product(xrange, yrange)

    if method == "Euler"
        for (x, y) in grid
            z = x + y*1im
            if abs(1+z) <= 1
                push!(conv, 1)
            else
                push!(conv, 0)
            end
        end
        axis_var = "z"
        annotation = (1, 1, "")


    elseif method == "Backwards Euler"
        for (x, y) in grid
            z = x + y*1im
            if 1/abs(1-z) <= 1
                push!(conv, 1)
            else
                push!(conv, 0)
            end
        end
        axis_var = "z"
        annotation = (1, 1, "")


    elseif method == "RK4"
        for (x, y) in grid
            z = x + y*1im
            if abs(1 + (z/24)*(24 + 12*z + 4*z^2 + z^3)) <= 1
                push!(conv, 1)
            else
                push!(conv, 0)
            end
        end
        axis_var = "z"
        annotation = (1, 1, "")


    elseif method == "RIDC Euler-Euler"
        a = 0           # Lower bound
        b = 50          # Upper bound
        u_init = 1      # Initial condition for IVP
        n = 10          # Number of intervals for the RIDC Euler-Euler method
        k = 5           # Number of time steps in each interval for the RIDC Euler-Euler method
        l = 3           # Number of corrections
        h = (b-a)/(n*k) # Stepsize

        # Get integration matrix from nodes
        W = lagrange_weights(a, b, n, k)

        for (x, y) in grid
            z = x + y*1im

            lambda = z/h    # Stability Test ODE Coefficient
            f(u, t) = lambda*u

            u_matrix = RIDC_Euler(f, a, b, u_init, W, n, k, l)
            
            if abs(u_matrix[end, end]) <= 1
                push!(conv, 1)
            else
                push!(conv, 0)
            end
        end
        axis_var = "λ"
        annotation = (1.55*xbounds[2], 0.5*ybounds[1], text("Step Size: $(round(h, digits = 2)) \nNo. Intervals: $n \nNo. Time Steps: $k \nNo. Corrections: $l", :black, :bottom, :8))


    elseif method == "RIDC RK2-RK2"
        a = 0           # Lower bound
        b = 50          # Upper bound
        u_init = 1      # Initial condition for IVP
        n = 10          # Number of intervals for the RIDC RK2-RK2 method
        k = 5           # Number of time steps in each interval for the RIDC RK2-RK2 method
        l = 3           # Number of corrections
        h = (b-a)/(n*k) # Stepsize

        # Get integration matrix from nodes
        W = lagrange_weights(a, b, n, k)

        for (x, y) in grid
            z = x + y*1im

            lambda = z/h    # Stability Test ODE Coefficient
            f(u, t) = lambda*u

            u_matrix = RIDC_RK2(f, a, b, u_init, W, n, k, l)
            
            if abs(u_matrix[end, end]) <= 1
                push!(conv, 1)
            else
                push!(conv, 0)
            end
        end
        axis_var = "λ"
        annotation = (1.55*xbounds[2], 0.5*ybounds[1], text("Step Size: $(round(h, digits = 2)) \nNo. Intervals: $n \nNo. Time Steps: $k \nNo. Corrections: $l", :black, :bottom, :8))


    end

    p = plot(title = "$method Method\nRegion of Stability\n", xlims = xbounds, ylims = ybounds, xlabel = "Re($axis_var)", ylabel = "Im($axis_var)", legend_position = :outertopright, aspect_ratio = :equal)
    contour!(xrange, yrange, conv, color = :black, fill = true, colorbar = false, fillalpha = [0.2, 0.3], fillcolor = [:blue, :red])
    vline!([0], linestyle = :dash, color = :black, label = "")
    hline!([0], linestyle = :dash, color = :black, label = "")
    plot!([0], [0], linecolor = "black", fill = (0, 0.2, :blue), label = "Search Space")
    plot!([0], [0], linecolor = "black", fill = (0, 0.8, :red), label = "Stable Region")
    annotate!([annotation])
    display(p)
    return(p)
end


Stability_Test()

Stability_Test(method = "Backwards Euler")

Stability_Test(method = "RK4", xbounds = [-4, 4], ybounds = [-4, 4], res = 1500)

Stability_Test(method = "RIDC Euler-Euler")

Stability_Test(method = "RIDC RK2-RK2")