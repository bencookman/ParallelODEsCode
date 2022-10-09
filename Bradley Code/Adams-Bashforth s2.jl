# Get packages if you don't already have them (uncomment and run the line below)
# using Pkg; Pkg.add("Plots"); Pkg.add("BenchmarkTools")

# Import packages
using Plots, BenchmarkTools

# ODE
# u'(t) = cos(t)*(sin(t) - u(t))

# Initial conditions
# t₀ = 0, u₀ = 1/3

# Exact solution
U(t) = 4/(3*exp(sin(t))) + sin(t) - 1

# Time domain [a, b]
a, b = [0, 10]

# Number of time steps
N = 1000000

# AB method (s=2) function
function AB_Method(a, b, N, disp)
    # Step size (h)
    h = (b-a)/N

    # AB method (s=2) calc
    t = a:h:b
    u = Vector{Float64}(undef, N+1)
    # We make the assumption that u[2] = u[1]
    u[1:2] .= 1/3

    for n in 1:N-1
        u[n+2] = u[n+1] + 0.5*h*(3*cos(t[n])*(sin(t[n]) - u[n]) - cos(t[n+1])*(sin(t[n+1]) - u[n+1]))
    end

    p = plot(t, U.(t), color = :green, labels = "True", title = "Adams-Bashforth Method s=2 (h = $h)") 
    plot!(t, u, color = :red, linestyle = :dash, xlabel = "t", ylabel = "u(t)", labels = "Approx")

    if disp == true 
        display(plot(p)) 
    end
    
    # Function to calculate absolute difference between two numbers
    d(x, y) = abs(x - y)

    errors = d.(U.(t), u)
    mean_err = mean(errors)
    max_err = maximum(errors)
    # Create error stats dict
    err_stats = Dict(:Errors => errors, :Mean => mean_err, :Max => max_err)
    return(err_stats)
end

# Displays true function and approximation on the same plot and gets error stats
err_stats = AB_Method(a, b, N, true)
# Print mean absolute error
println(err_stats[:Mean])
# Print max absolute error
println(err_stats[:Max])

# While benchmarking, the function is run many times and so displaying plots is disabled
@benchmark AB_Method(a, b, N, false)