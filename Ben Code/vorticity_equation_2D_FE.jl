using FFTW

""" UNFINISHED """
function vorticity_equation_spectral_FE(;
    nu=0.1,
    t_e=10.0,
    N_space=2^8,
    N_t=10_000
)
    t_coords = range(0.0, t_e, N_t)
    Δt = t[2] - t[1]
    Δspace = 2π/N_space
    x_coords = range(Δspace, 2π, N_space)
    y_coords = range(Δspace, 2π, N_space)
    k_coords = fftfreq(N_space, N_space)

    # Initialise 2D meshes
    x = [x_value for x_value in x_coords, y_value in y_coords]
    y = [y_value for x_value in x_coords, y_value in y_coords]
    k_x = [k_x_value for k_x_value in k_coords, k_y_value in k_coords]
    k_y = [k_y_value for k_x_value in k_coords, k_y_value in k_coords]
    k² = k_x^2 + k_y^2
    u = Array{Float64, 2}(undef, N_space, N_space)
    v = Array{Float64, 2}(undef, N_space, N_space)
    w = Array{Float64, 2}(undef, N_space, N_space)
    u_fft = Array{ComplexF64{Float64}, 2}(undef, N_space, N_space)
    v_fft = Array{ComplexF64{Float64}, 2}(undef, N_space, N_space)
    w_fft = Array{ComplexF64{Float64}, 2}(undef, N_space, N_space)

    # Impose initial conditions
    # Initialise w to curl of flow

    # Set up return arrays (mostly for plotting after the fact)
    u_return = [u]
    v_return = [v]
    w_return = [w]

    # Time step!
    function f(nu, k_x, k_y, u, v, w_fft)
        u_conv = fft(u.*real(ifft(im.*k_x.*w_fft)))
        v_conv = fft(v.*real(ifft(im.*k_y.*w_fft)))
        k² = k_x^2 + k_y^2
        return - u_conv - v_conv - nu.*k².*w_fft
    end
    @showprogress "Time stepping..." for _ in t_coords[1:end - 1]
        # Forward Euler time step
        w_fft .+= Δt.*f(nu, k_x, k_y, u, v, w_fft)

        w = real(ifft(w_fft))
        u = real(ifft(u_fft))   # How do we retrieve these from vorticity information?
        v = real(ifft(v_fft))
        push!(w_return, w)
        push!(u_return, u)
        push!(v_return, v)
    end

    return (t_coords, x_coords, y_coords, u_return, v_return, w_return)
end