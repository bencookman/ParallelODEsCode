module ProjectTools

using LinearAlgebra, Statistics, PyCall

export
    err_abs,
    err_cum,
    err_norm,
    err_max,
    integration_matrix_equispaced

err_abs(exact, approx) = abs.(exact - approx)
err_cum(exact, approx) = [sum(err_abs(exact[1:i], approx[1:i])) for i in 1:length(data_correct)]
err_norm(exact, approx, p) = norm(err_abs(exact, approx), p)
err_max(exact, approx)  = maximum(err_abs(exact, approx))

function integration_matrix_equispaced(M)
    scipy_interpolate = pyimport("scipy.interpolate")

    x = collect(0.0:M)
    # Calculate anti-derivatives of polynomial interpolants
    q_func = []
    for i in 1:(M + 1)
        yᵢ = zeros(Float64, M + 1)
        yᵢ[i] = 1.0
        pᵢ_coeffs = scipy_interpolate.lagrange(x, yᵢ).c |> reverse              # Lagrange interpolant to yᵢ at x
        qᵢ_coeffs = pᵢ_coeffs ./ collect(1.0:length(pᵢ_coeffs))                 # Anti-derivative of pᵢ
        qᵢ_func(t) = sum(qᵢⱼ*t^Float64(j) for (j, qᵢⱼ) in enumerate(qᵢ_coeffs))
        push!(q_func, qᵢ_func)
    end
    # Use anti-derivatives to evaluate integration matrix
    return [qᵢ_func(m + 1) - qᵢ_func(m) for m in 0.0:(M - 1), qᵢ_func in q_func]
end

end