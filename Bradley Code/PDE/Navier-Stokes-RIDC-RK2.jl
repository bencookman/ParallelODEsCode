using Plots
using ProgressMeter
using FFTW
using LinearAlgebra

cd(@__DIR__)

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

# PDE
# 2D Incompressible Navier Stokes (constant density ρ₀)
# ∂u/∂t + (u·∇)u - μ/ρ₀ ∇²u = -∇(p/ρ₀) + f

# 2D Vorticity Equation (kinematic viscosity v)
# ∂w/∂t = v ∇²w - (u·∇)w + ∇ x f

# Spatial Domain
# (x, y) ∈ {(x, y) : -π <= x <= π, -π <= y <= π}

# Time Domain
# t ∈ [0, ∞)

# Initial Condition
# u(x, y, 0) = 0    Fluid is initially at rest

# Boundary Conditions
# Periodic boundary with period 2π

# Pseudo-Spectral Method Solution (Heun's Method Time Integration)

N_POINTS = 100
KINEMATIC_VISCOSITY = 0.5
MAX_TIME = 2
N_TIME_INTERVALS = 100
N_TIME_STEPS = 5
N_CORRECTIONS = 1

function main()
    element_length = 2*pi / (N_POINTS - 1)
    time_step_length = MAX_TIME / (N_TIME_INTERVALS * N_TIME_STEPS)

    x_range = range(-pi, pi, length = N_POINTS)
    y_range = range(-pi, pi, length = N_POINTS)

    # Get discretised spatial grid
    coordinates_x = [x for x in x_range, y in y_range]
    coordinates_y = [y for x in x_range, y in y_range]

    # Get wavenumbers in spatial grid format and their norms
    wavenumbers_1d = fftfreq(N_POINTS) .* N_POINTS
    wavenumbers_x = [kx for kx in wavenumbers_1d, ky in wavenumbers_1d]
    wavenumbers_y = [ky for kx in wavenumbers_1d, ky in wavenumbers_1d]
    wavenumbers_norm = [norm([kx, ky]) for kx in wavenumbers_1d, ky in wavenumbers_1d]

    wavenumbers_norm[iszero.(wavenumbers_norm)] .= 1
    wavenumbers_normalised_x = wavenumbers_x ./ wavenumbers_norm
    wavenumbers_normalised_y = wavenumbers_y ./ wavenumbers_norm

    # Define body force which acts only in x direction
    force_x = 10 * (
        exp.(-5 * ((coordinates_x .+ 0.5*pi).^2 + (coordinates_y .+ 0.2*pi).^2))
        -
        exp.(-5 * ((coordinates_x .- 0.5*pi).^2 + (coordinates_y .- 0.2*pi).^2))
    )

    # Get fourier transform of body force
    force_x_fft = fft(force_x)

    # Initialise Newton-Cotes integration weights matrix
    W = lagrange_weights(0:time_step_length:MAX_TIME/N_TIME_INTERVALS)

    # Initialise list of arrays
    # Dims: (x, y, time, corrections) will be overwritten at each time interval
    velocities_x = zeros(N_POINTS, N_POINTS, N_TIME_STEPS+1, N_CORRECTIONS+1)
    velocities_y = zeros(N_POINTS, N_POINTS, N_TIME_STEPS+1, N_CORRECTIONS+1)

    vorticities = zeros(N_POINTS, N_POINTS, N_TIME_STEPS+1, N_CORRECTIONS+1)
    vorticities_fft = zeros(ComplexF64, N_POINTS, N_POINTS, N_TIME_STEPS+1, N_CORRECTIONS+1)
    
    # Define function for calculating the time derivative of the fourier transformed vorticity (f) for current time step using past data
    function dw__dt_fft(vort_fft_prev, velo_x_prev, velo_y_prev, interval, i)
        # Initialise time derivative of the fourier transformed vorticity for current time step
        dw_dt_fft = zero(coordinates_x)

        # Apply convection term in fourier space
        dw_dt_fft -= (
            fft(velo_x_prev .* ifft(1im * wavenumbers_x .* vort_fft_prev))
            +
            fft(velo_y_prev .* ifft(1im * wavenumbers_y .* vort_fft_prev))
        )

        # Apply a mask which sets to 0 the values of the non-linear convection term with large wavenumbers
        mask_wavenumbers = (abs.(wavenumbers_1d) .>= 2*fld(N_POINTS, 3))
        dw_dt_fft[mask_wavenumbers, mask_wavenumbers] .= 0

        # Apply curl of body force term in fourier space during the first time step of a unit of time only
        time_current = (interval - 1) * N_TIME_STEPS * time_step_length + (i - 1) * time_step_length
        pre_factor = max(1 - time_current, 0)
        dw_dt_fft -= pre_factor * 1im * wavenumbers_y .* force_x_fft

        # Apply diffusion term in fourier space
        dw_dt_fft -= KINEMATIC_VISCOSITY * wavenumbers_norm .* vort_fft_prev

        return dw_dt_fft
    end

    # Initialise animation of vorticity
    anim = Plots.Animation()

    @showprogress "Time Stepping..." for interval in 1:N_TIME_INTERVALS

        #------------------------------------------------------------------------------------------#
        # RK2 (Heun's Method) Prediction Step for Time Interval

        for i in 1:N_TIME_STEPS
    
            # Get time derivative of the fourier transformed vorticity for current time step
            vort__dt_fft_now = dw__dt_fft(
                vorticities_fft[:, :, i, 1],
                velocities_x[:, :, i, 1], 
                velocities_y[:, :, i, 1],
                interval,
                i
            )

            # Approximate fourier transformed vorticity at next time step using Euler method
            vorticities_fft_next = vorticities_fft[:, :, i, 1] + time_step_length * vort__dt_fft_now

            # Approximate velocities at next time step
            velocities_x_next = real(ifft(  1im * wavenumbers_normalised_y .* vorticities_fft_next))
            velocities_y_next = real(ifft(- 1im * wavenumbers_normalised_x .* vorticities_fft_next))

            # Get approx time derivative of the fourier transformed vorticity for next time step
            vort__dt_fft_next = dw__dt_fft(
                vorticities_fft_next,
                velocities_x_next,
                velocities_y_next,
                interval,
                i+1
            )

            # RK2 method prediction step (Find value of fourier transform of vorticity at next time step)
            vorticities_fft[:, :, i+1, 1] = (
                vorticities_fft[:, :, i, 1] + 0.5 * time_step_length * (vort__dt_fft_now + vort__dt_fft_next)
            )

            # Get new values for vorticity
            vorticities[:, :, i+1, 1] = real(ifft(vorticities_fft[:, :, i+1, 1]))

            # Get new values for components of velocity
            velocities_x[:, :, i+1, 1] = real(ifft(  1im * wavenumbers_normalised_y .* vorticities_fft[:, :, i+1, 1]))
            velocities_y[:, :, i+1, 1] = real(ifft(- 1im * wavenumbers_normalised_x .* vorticities_fft[:, :, i+1, 1]))
        end

        #------------------------------------------------------------------------------------------#
        # RK2 (Heun's Method) Correction Steps for Time Interval

        for k in 1:N_CORRECTIONS
            # Get list over the times in the current correction, of the time derivative of fft vorticity
            vort__dt_fft_vec = [dw__dt_fft(
                vorticities_fft[:, :, i, k],
                velocities_x[:, :, i, k],
                velocities_y[:, :, i, k],
                interval,
                i
            ) for i in 1:N_TIME_STEPS+1]

            # Newton-Cotes integration on the time derivative vector
            vort_fft_int = zeros(ComplexF64, N_POINTS, N_POINTS, N_TIME_STEPS)

            for m in 1:N_POINTS, n in 1:N_POINTS
                vec = [vort__dt_fft_vec[i][m, n] for i in 1:N_TIME_STEPS+1]
                vort_fft_int[m, n, :] = W * vec
            end

            for i in 1:N_TIME_STEPS

                # Get time derivative of fourier transformed vorticity (next correction level, current time)
                vort__dt_fft_next_correction = dw__dt_fft(
                    vorticities_fft[:, :, i, k+1],
                    velocities_x[:, :, i, k+1],
                    velocities_y[:, :, i, k+1],
                    interval,
                    i
                )

                # Get time derivative of fourier transformed vorticity (current correction level, next time)
                vort__dt_fft_next_time = dw__dt_fft(
                    vorticities_fft[:, :, i+1, k],
                    velocities_x[:, :, i+1, k],
                    velocities_y[:, :, i+1, k],
                    interval,
                    i+1
                )

                K1 = time_step_length * (vort__dt_fft_next_correction - vort__dt_fft_vec[i])

                # Approximate fourier transformed vorticity at next time step, next correction level
                vorticities_fft_next = vorticities_fft[:, :, i, k+1] + K1 + vort_fft_int[:, :, i]

                # Approximate velocities at next time step, next correction level
                velocities_x_next = real(ifft(  1im * wavenumbers_normalised_y .* vorticities_fft_next))
                velocities_y_next = real(ifft(- 1im * wavenumbers_normalised_x .* vorticities_fft_next))
                
                vort__dt_fft_rk2 = dw__dt_fft(
                    vorticities_fft_next,
                    velocities_x_next,
                    velocities_y_next,
                    interval,
                    i+1
                )

                K2 = time_step_length * (vort__dt_fft_rk2 - vort__dt_fft_next_time)

                # Find next value in time of fourier transformed vorticity for next correction step
                vorticities_fft[:, :, i+1, k+1] = (
                    vorticities_fft[:, :, i, k+1] + 0.5*K1 + 0.5*K2 + vort_fft_int[:, :, i]
                )

                # Get new values for vorticity
                vorticities[:, :, i+1, k+1] = real(ifft(vorticities_fft[:, :, i+1, k+1]))

                # Get new values for components of velocity
                velocities_x[:, :, i+1, k+1] = real(ifft(  1im * wavenumbers_normalised_y .* vorticities_fft[:, :, i+1, k+1]))
                velocities_y[:, :, i+1, k+1] = real(ifft(- 1im * wavenumbers_normalised_x .* vorticities_fft[:, :, i+1, k+1]))
            end
        end

        #------------------------------------------------------------------------------------------#
        # Set value of initial condition for new interval to be the last correction in previous interval

        vorticities[:, :, 1, :] = repeat(vorticities[:, :, end, end], 1, 1, N_CORRECTIONS+1)
        vorticities_fft[:, :, 1, :] = repeat(vorticities_fft[:, :, end, end], 1, 1, N_CORRECTIONS+1)

        velocities_x[:, :, 1, :] = repeat(velocities_x[:, :, end, end], 1, 1, N_CORRECTIONS+1)
        velocities_y[:, :, 1, :] = repeat(velocities_y[:, :, end, end], 1, 1, N_CORRECTIONS+1)

        #------------------------------------------------------------------------------------------#
        # Visualise Simulation for Time Interval (every 5 time-intervals and last)

        if mod(interval, 5) == 1 | (interval == N_TIME_INTERVALS)
            # Visualise velocity vector field
            display(quiver(reshape(coordinates_x[1:fld(N_POINTS, 100):end, 1:fld(N_POINTS, 100):end], :), reshape(coordinates_y[1:fld(N_POINTS, 100):end, 1:fld(N_POINTS, 100):end], :), quiver = (reshape(velocities_x[1:fld(N_POINTS, 100):end, 1:fld(N_POINTS, 100):end, end, end]/100, :), reshape(velocities_y[1:fld(N_POINTS, 100):end, 1:fld(N_POINTS, 100):end, end, end]/100, :)), markersize = 1, aspect_ratio = :equal, size = (680, 650)))


            # Visualise curl/vorticity 
            # Vorticity' will cause the plot to be interpreted as an image oriented correctly
            display(heatmap(x_range, y_range, vorticities[:, :, end, end]', c=:diverging_bkr_55_10_c35_n256, aspect_ratio = :equal, size = (680, 650)))
            frame(anim, heatmap(x_range, y_range, vorticities[:, :, end, end]', c=:diverging_bkr_55_10_c35_n256, aspect_ratio = :equal, size = (680, 650)))
        end
    end
    display(gif(anim, "2D-Navier-Stokes-RIDC-RK2.gif", fps = 2))
end

main()
