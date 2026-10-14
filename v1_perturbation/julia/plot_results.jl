# Optional plots from the CSV files written by run_tests.jl.  Requires Plots.jl:
#   julia -e 'using Pkg; Pkg.add("Plots")'
#   julia plot_results.jl
using DelimitedFiles, Plots

# ---- convergence + eps sweep
C, _ = readdlm("convergence.csv", ',', header = true)
p1 = plot(xscale = :log10, yscale = :log10, xlabel = "h [m]", ylabel = "L2 error in p'",
          title = "Convergence, periodic normal mode", legend = :bottomright, legendfontsize = 6)
for key in unique([(C[i, 1], C[i, 2], C[i, 3], C[i, 4]) for i in 1:size(C, 1)])
    rows = [i for i in 1:size(C, 1) if (C[i, 1], C[i, 2], C[i, 3], C[i, 4]) == key]
    h = 2000.0 ./ Float64.(C[rows, 5])
    plot!(p1, h, Float64.(C[rows, 6]), marker = :circle, label = "$(key[1]) $(key[2]) eps=$(key[3]) p=$(key[4])")
end

E, _ = readdlm("eps_sweep.csv", ',', header = true)
E = Float64.(E)
p2 = plot(E[:, 1], E[:, 2], xscale = :log10, yscale = :log10, marker = :circle, label = "dirk",
          xlabel = "eps c / h", ylabel = "L2 error in p'", title = "Relaxation error, Ne=16, p=3")
plot!(p2, E[:, 1], E[:, 3], marker = :square, ls = :dash, label = "imex")
hline!(p2, [E[1, 4]], color = :black, ls = :dot, label = "eps = 0 (HDG)")
savefig(plot(p1, p2, layout = (1, 2), size = (1200, 450)), "convergence_julia.png")

# ---- wall demo
P, _ = readdlm("wall_demo_profile.csv", ',', header = true)
Hs, _ = readdlm("wall_demo_history.csv", ',', header = true)
q1 = plot(xlabel = "p' [Pa] at t = 40 s", ylabel = "z [km]", legendfontsize = 6)
q2 = plot(xlabel = "t [s]", ylabel = "|mass drift|", yscale = :log10, legend = false)
q3 = plot(xlabel = "t [s]", ylabel = "trace residual", yscale = :log10, legend = false)
for lab in unique(P[:, 1])
    r = P[:, 1] .== lab
    plot!(q1, Float64.(P[r, 5]), Float64.(P[r, 2]) ./ 1000, label = lab)
    r = Hs[:, 1] .== lab
    plot!(q2, Float64.(Hs[r, 2]), abs.(Float64.(Hs[r, 3])) .+ 1e-30)
    plot!(q3, Float64.(Hs[r, 2]), Float64.(Hs[r, 4]) .+ 1e-30)
end
savefig(plot(q1, q2, q3, layout = (1, 3), size = (1500, 420)), "wall_demo_julia.png")
println("wrote convergence_julia.png, wall_demo_julia.png")
