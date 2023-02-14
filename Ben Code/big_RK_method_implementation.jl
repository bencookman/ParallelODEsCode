using Dates, LaTeXStrings, Plots, Dates


"""
Implementation of the Cooper-Verner eighth order Runge-Kutta method, written as
flatly as I have the energy for.

Use y instead of η here for conceptual simplicity.
"""
function RK8_Cooper_Verner_solve(
    f,
    t_s,
    t_e,
    y_s,
    N
)
    t = range(t_s, t_e, N + 1) |> collect
    Δt = (t_e - t_s)/N
    y = zeros(typeof(y_s), N + 1)
    y[1] = y_s

    sq = sqrt(21)
    c = Float64[
        0.0, 0.5, 0.5,
        (7 + sqrt(21))/14, (7 + sqrt(21))/14, 0.5,
        (7 - sqrt(21))/14, (7 - sqrt(21))/14, 0.5,
        (7 + sqrt(21))/14, 1.0
    ]
    # a = Float64[
    #     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    #     0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    #     0.25 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    #     1/7 (-7 - 3sqrt(21))/98 (21 + 5sqrt(21))/49 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    #     (11 + sqrt(21))/84 0.0 (18 + 4sqrt(21))/63 (21 - sqrt(21))/252 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    #     (5 + sqrt(21))/48 0.0 (9 + sqrt(21))/36 (-231 + 14sqrt(21))/360 (63 - 7sqrt(21))/80 0.0 0.0 0.0 0.0 0.0 0.0;
    #     (10 - sqrt(21))/42 0.0 (-432 + 92sqrt(21))/315 (633 - 145sqrt(21))/90 (-504 + 115sqrt(21))/70 (63 - 13sqrt(21))/35 0.0 0.0 0.0 0.0 0.0;
    #     1/14 0.0 0.0 0.0 (14 - 3sqrt(21))/126 (13 - 3sqrt(21))/63 1/9 0.0 0.0 0.0 0.0;
    #     1/32 0.0 0.0 0.0 (91 - 21sqrt(21))/576 11/72 (-385 - 75sqrt(21))/1152 (63 + 13sqrt(21))/128 0.0 0.0 0.0;
    #     1/14 0.0 0.0 0.0 1/9 (-733 - 147sqrt(21))/2205 (515 + 111sqrt(21))/504 (-51 - 11sqrt(21))/56 (132 + 28sqrt(21))/245 0.0 0.0;
    #     0.0 0.0 0.0 0.0 (-42 + 7sqrt(21))/18 (-18 + 28sqrt(21))/45 (-273 - 53sqrt(21))/72 (301 + 53sqrt(21))/72 (28 - 28sqrt(21))/45 (49 - 7sqrt(21))/18 0.0
    # ]
    a = zeros(Float64, 11, 11)
    a[2,1]=1/2
    a[3,1]=1/4
    a[3,2]=1/4
    a[4,1]=1/7
    a[4,2]=-1/14-3/98*sq
    a[4,3]=3/7+5/49*sq
    a[5,1]=11/84+1/84*sq
    a[5,2]=0
    a[5,3]=2/7+4/63*sq
    a[5,4]=1/12-1/252*sq
    a[6,1]=5/48+1/48*sq
    a[6,2]=0
    a[6,3]=1/4+1/36*sq
    a[6,4]=-77/120+7/180*sq
    a[6,5]=63/80-7/80*sq
    a[7,1]=5/21-1/42*sq
    a[7,2]=0
    a[7,3]=-48/35+92/315*sq
    a[7,4]=211/30-29/18*sq
    a[7,5]=-36/5+23/14*sq
    a[7,6]=9/5-13/35*sq
    a[8,1]=1/14
    a[8,2]=0
    a[8,3]=0
    a[8,4]=0
    a[8,5]=1/9-1/42*sq
    a[8,6]=13/63-1/21*sq
    a[8,7]=1/9
    a[9,1]=1/32
    a[9,2]=0
    a[9,3]=0
    a[9,4]=0
    a[9,5]=91/576-7/192*sq
    a[9,6]=11/72
    a[9,7]=-385/1152-25/384*sq
    a[9,8]=63/128+13/128*sq
    a[10,1]=1/14
    a[10,2]=0
    a[10,3]=0
    a[10,4]=0
    a[10,5]=1/9
    a[10,6]=-733/2205-1/15*sq
    a[10,7]=515/504+37/168*sq
    a[10,8]=-51/56-11/56*sq
    a[10,9]=132/245+4/35*sq
    a[11,1]=0
    a[11,2]=0
    a[11,3]=0
    a[11,4]=0
    a[11,5]=-7/3+7/18*sq
    a[11,6]=-2/5+28/45*sq
    a[11,7]=-91/24-53/72*sq
    a[11,8]=301/72+53/72*sq
    a[11,9]=28/45-28/45*sq
    a[11,10]=49/18-7/18*sq

    for i in 1:N
        k = [f(t[i], y[i])]
        for r in 2:11
            k_sum = 0.0
            for l in 1:(r - 1)
                k_sum += a[r, l]*k[l]
            end
            push!(k, f(t[i] + c[r]*Δt, y[i] + Δt*k_sum))
        end
        y[i + 1] = y[i] + k[1]*(1/20) + k[8]*(49/180) + k[9]*(16/45) + k[10]*(49/180) + k[11]*(1/20)
    end

    return (t, y)
end

function test_big_RK_method()
    # f(t, y) = 4t*sqrt(y)
    # t_e = 5.0
    # y_s = 1.0 + 0.0im
    # y(t) = (1 + t^2)^2
    f(t, y) = (y - 2t*y^2)/(1 + t)
    t_e = 1.0
    y_s = 0.4
    y(t) = (1 + t)/(t^2 + 1/0.4)

    N_array = 10:3:100
    orders_to_plot = [8]

    err_array = []
    for N in N_array
        (_, y_approx) = RK8_Cooper_Verner_solve(f, 0.0, t_e, y_s, N)
        err = abs(y(t_e) - y_approx[end])
        push!(err_array, err)
    end

    Δt_array = t_e./N_array
    plot_err = plot(
        xscale=:log10, yscale=:log10, xlabel=L"Δt", ylabel="||E||",
        key=:bottomright, size=(1600, 1200), thickness_scaling=2.0
    )
    plot!(
        plot_err,
        Δt_array, err_array,
        markershape=:square, label="Approximation using RK8 Cooper-Verner method", color = :red,
    )
    for order in orders_to_plot
        err_order_array = Δt_array.^order # Taking error constant = 1 always
        plot!(
            plot_err, Δt_array, err_order_array,
            linestyle=:dash, label=L"1\cdot (\Delta t)^%$order"
        )
    end
    # dtstring = Dates.format(now(), "DY-m-d-TH-M-S")
    # fname = "Ben Code/output/convergence/$dtstring-convergence-IDC_FE,SDC_FE.png"
    # savefig(plot_err, fname)
    display(plot_err)


end