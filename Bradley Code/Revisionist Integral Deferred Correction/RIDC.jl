using Plots

function lagrange_weights(nodes)
    k = length(nodes)
    weights = zeros(Float64, k-1, k)
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

function RICD(f, a, b, n, k, u_init, l)
    big_h = (b-a)/n
    h = big_h/k

    nodes = [a+h*(i-1) for i in 1:k+1]
    # Get integration weights matrix for nodes
    W = lagrange_weights(nodes)

    u_full_matrix = zeros(Float64, l+1, n*k+1)
    for p in 1:n
        nodes = [a+big_h*(p-1)+h*(i-1) for i in 1:k+1]

        # Initial prediction via Euler Method
        u_hat = zeros(Float64, k+1)
        u_hat[1] = u_init
        for i in 1:k
            u_hat[i+1] = u_hat[i] + h*f(u_hat[i], nodes[i])
        end

        # Initialise correction matrix with its initial conditions
        u_matrix = zeros(Float64, l+1, k+1)
        u_matrix[1, :] = u_hat
        u_matrix[:, 1] .= u_init

        # Corrections section via integral deferred correction
        for l in 1:l
            f_hat = f.(u_matrix[l, :], nodes)
            f_int = W * f_hat
            errors = zeros(Float64, k+1)
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

f(u, t) = -2*t*u

U(t) = exp(-t^2)

a = 0
b = 2
n = 10
k = 5
u_init = 1
l = 5

u_matrix = RICD(f, a, b, n, k, u_init, l)
tt = a:1e-3:b
plot(tt, U)
plot!(a:(b-a)/(n*k):b, u_matrix[end, :])

