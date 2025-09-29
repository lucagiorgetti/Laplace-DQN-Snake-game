#plot_traj.jl
###########################################################################
#computes the SVD of the deviation matrix and plots the trajectory in the 2 most important directions
###########################################################################
include("imports.jl")

BSON.@load "./D_matrices/D_very_long_double_training3.bson" deviation_matrix

#SVD decomposition
F = svd(deviation_matrix)
U, S, V = F

#Now building the Expected Spectral distribution of 1/(K-1) D D^T

K = size(deviation_matrix)[2]
lambda = S.^2 /(K - 1)

eig_pos = filter(>(1e-7), lambda) 

# make log-spaced bins between min and max eigenvalues
logbins = exp10.(range(log10(minimum(eig_pos)), ceil(log10(maximum(eig_pos))), length=50))

pl = histogram(
    eig_pos,
    bins = logbins,
    color = "red",
    xlabel = "Eigenvalues (log scale)",
    ylabel = "Counts (log scale)",
    #title = "Correlation Matrix Spectrum",
    legend = false,
    normalize = false,
    xaxis = :log10,
    yaxis = :log10,
    xlim = extrema(logbins),
    xtickfont = font(13),
    ytickfont = font(13),
    xguidefont = font(15),
    yguidefont = font(15),
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
    right_margin = 10mm
)
xticks!(pl, 10.0 .^(floor(Int,log10(minimum(eig_pos))):ceil(Int,log10(maximum(eig_pos)))))
savefig(pl, "correlation_histo.png")

#now I need to find n_cols that accounts for the 99 per cent of variance of 1/(K-1) D D^T
function compute_n_cols(eigs::Vector)
    tot = sum(eigs)
    lim = 99/100 * tot
    n_cols = 0
    cum = 0

    for l in lambda
        cum += l
        n_cols += 1
        if cum >= lim
            break
        end
    end

    @info "The number of columns of D to keep is $n_cols"
    return n_cols
end

compute_n_cols(lambda)

#now I want a plot of the columns of the deviation matrix along the two most important directions
proj = U[:,1:2]
Y = transpose(proj) * deviation_matrix

n = size(Y, 2)
idx = 1:n

pl_series = plot(
    idx, Y[1, :],
    lw = 4,
    xlabel = "Time step",
    ylabel = "1st",
    legend = false,
    title = "Time series of first two singular directions",
    layout = (2, 1),   # two rows, one column
    xtickfont = font(13),
    ytickfont = font(13),
    xguidefont = font(15),
    yguidefont = font(15),
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
    right_margin = 10mm,
    yticks = -5:5:5
)

plot!(pl_series[2], idx, Y[2, :],
    lw = 4, 
    xlabel = "Time Step",
    ylabel = "2nd",
    legend = false,
    xtickfont = font(13),
    ytickfont = font(13),
    xguidefont = font(15),
    yguidefont = font(15),
    left_margin = 10mm,
    bottom_margin = 10mm,
    top_margin = 10mm,
    right_margin = 10mm,
    yticks = -3:3:3
)

savefig(pl_series, "trajectory_series.png")

