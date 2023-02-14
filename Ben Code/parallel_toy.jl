using
    ProgressMeter,
    BenchmarkTools,
    Dates,
    Plots


function calculate_fibonacci_parallel()
    N = 10
    number_array = zeros(N + 1, 2)
    number_array[1, :] .= 1

    for i in 1:N
        Threads.@threads :static for l in 1:2
            if l == 1
                number_array[i + 1, 1] = number_array[i, 1] + number_array[i, 2]
            end
            if l == 2
                number_array[i + 1, 2] = number_array[i, 1]
            end
        end
    end
    number_array
end

function calculate_fibonacci()
    N = 10
    number_array = zeros(N + 1, 2)
    number_array[1, :] .= 1

    for i in 1:N
        number_array[i + 1, 1] = number_array[i, 1] + number_array[i, 2]
        number_array[i + 1, 2] = number_array[i, 1]
    end
    number_array
end

function calculate_parallel()
    N = 100
    number_array = zeros(N + 1, Threads.nthreads())
    number_array[1, :] .= 1

    for i in 1:N
        Threads.@threads for l in 1:Threads.nthreads()
            number_array[i + 1, l] = number_array[i, 1] + number_array[i, 2]
        end
    end
    number_array
end

function calculate()
    N = 100
    number_array = zeros(N + 1, 2)
    number_array[1, :] .= 1

    for i in 1:N
        for l in 1:2
            number_array[i + 1, l] = number_array[i, 1] + number_array[i, 2]
        end
    end
    number_array
end

function RIDC_FE_parallel(
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
        ## STARTUP
        for m in 1:(2M - 1)
            k = j*K + m
            η[k + 1, 1] = η[k, 1] + Δt*f(t[k], η[k, 1])
        end
        for l in 2:p
            for m in 1:M
                k = j*K + m
                I = (j*K + 1):(j*K + M + 1)
                ∫fₖ = dot(S[m, :], f.(t[I], η[I, l - 1]))
                η[k + 1, l] = η[k, l] + Δt*(f(t[k], η[k]) - f(t[k], η[k, l - 1])) + Δt*∫fₖ
            end
            for m in (M + 1):(2M - l)
                k = j*K + m
                I = (k + 1 - M):(k + 1)
                ∫fₖ = dot(S[M, :], f.(t[I], η[I, l - 1]))
                η[k + 1, l] = η[k, l] + Δt*(f(t[k], η[k, l]) - f(t[k], η[k, l - 1])) + Δt*∫fₖ
            end
        end

        ## IN PIPELINE
        for m in M:(K - p + 1)
            Threads.@threads :static for l in 1:p
                k = j*K + m - l + p
                if l == 1
                    η[k + 1, 1] = η[k, 1] + Δt*f(t[k], η[k, 1])
                else
                    I = (k + 1 - M):(k + 1)
                    ∫fₖ = dot(S[M, :], f.(t[I], η[I, l - 1]))
                    η[k + 1, l] = η[k, l] + Δt*(f(t[k], η[k, l]) - f(t[k], η[k, l - 1])) + Δt*∫fₖ
                end
            end
        end

        ## TERMINATION
        k = (j + 1)*K
        η[k + 1, 1] = η[k, 1] + Δt*f(t[k], η[k, 1])
        for l in 2:p
            for m in ():(2M - l)
                k = j*K + m
                I = (k + 1 - M):(k + 1)
                ∫fₖ = dot(S[M, :], f.(t[I], η[I, l - 1]))
                η[k + 1, l] = η[k, l] + Δt*(f(t[k], η[k, l]) - f(t[k], η[k, l - 1])) + Δt*∫fₖ
            end
        end

    end

    return (t, η)
end
