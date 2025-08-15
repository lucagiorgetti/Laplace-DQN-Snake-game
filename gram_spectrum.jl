include("imports.jl")

struct MarchenkoPastur
    gamma::Real
    MarchenkoPastur(gamma) = 0 < gamma <= 1 ? new(gamma) : error("Gamma must be in (0, 1]")
end

function minimum(d::MarchenkoPastur)
    (1 - sqrt(d.gamma))^2
end

function maximum(d::MarchenkoPastur)
    (1 + sqrt(d.gamma))^2
end

function pdf(d::MarchenkoPastur, x::Real)
    lambdamin = minimum(d)
    lambdamax = maximum(d)

    if lambdamin < x < lambdamax
        return sqrt((lambdamax - x) * (x - lambdamin))/(d.gamma * x * 2 * pi)
    else
        return 0
    end
end

BSON.@load "./gram_matrix/very_long_double_training3.bson" gram_matrix

eigs = eigvals(gram_matrix)

"""
eigs_dict = countmap(eigs)
eigs_vals = keys(eigs_dict)
eigs_counts = values(eigs_dict)
"""
tr = Trainer()
param_count = length(Flux.destructure(tr.model.q_net)[1])
K = size(gram_matrix)[1]
gamma = K/param_count

dist =  MarchenkoPastur(gamma)

pl = histogram(eigs,
        bins = 100,
        xlabel = "Eigenvalues",
        ylabel = "Counts",
        title = "Correlation Matrix Spectrum",
        legend = false,
        color = "red", 
        normed = true,
        lims = (0,100)
        )
        
plot!(x->pdf(dist,x))

display(pl)
