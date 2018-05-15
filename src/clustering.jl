using Clustering
using Logging

struct STDPConfig
    n_hidden::Int64
    dt::Float64
    weight_mean::Float64
    weight_std::Float64
    t_train::Int64
    t_blank::Int64
    fr::Float64
    pre_dt::Float64
    pre_inc::Float64
    pre_target::Float64
    vstart::Float64
    vthresh::Float64
    vscale::Float64
    wmax::Float64
    stdp_lr::Float64
    stdp_mu::Float64
    inhib_weight::Float64
    neuron_function::Function
end

include("network.jl")
include("neural_functions.jl")

"""
STDP as a clustering function. X is unlabeled data and n is the number of
clusters. X must be between 0 and 1. Returns an Array{Int64} of labels with
size(X,2)
"""
function stdp_cluster(X::Array{Float64}, Y::Array{Int64}, n_cluster::Int64, nfunc::Function;
                      seed=10, logfile="stdp.log", problem="iris", fname="lif",
                      train_epochs=1, n_hidden=2*size(X, 1), dt=0.001,
                      weight_mean=0.5, weight_std=0.1, t_train=350, t_blank=150,
                      fr=65.0, pre_dt=20.0, pre_inc=1.0, pre_target=0.4,
                      vstart=-65.0, vthresh=30.0, vscale=100.0, wmax=1.0,
                      stdp_lr=0.0001, stdp_mu=2.0,
		      ika=0.02, ikb=0.2, ikd=2.0,
                      inhib_weight=0.1)::Array{Int64}

    vstart = vstart / vscale; vthresh = vthresh / vscale

    if fname == "lif"
        nfunc = i->lif(i, vstart, vthresh)
    elseif fname == "izhikevich"
        nfunc = i->izhikevich(i, vstart, vthresh, vscale;
				 a=ika, b=ikb, c=vstart/vscale, d=ikd/vscale)
    elseif fname == "fhn"
        nfunc = i->fhn(i, vstart, vthresh)
    end

    cfg = STDPConfig(n_hidden, dt, weight_mean, weight_std, t_train, t_blank,
                     fr, pre_dt, pre_inc, pre_target, vstart, vthresh, vscale,
                     wmax, stdp_lr, stdp_mu, inhib_weight, nfunc)

    rng = MersenneTwister(seed)
    n_input = size(X, 1)
    network = Network(n_input, n_hidden, n_cluster, cfg; rng=rng)

    # training
    for epoch in 1:train_epochs
        labels = iterate!(network, X, cfg, true, rng=rng)
        acc = randindex(Y, labels)
        Logging.info(@sprintf("A: %d %d %s %s %0.4f %0.4f %0.4f %0.4f",
                              epoch, seed, problem, fname, acc[1], acc[2],
                              acc[3], acc[4]))
    end

    # testing
    final_labels = iterate!(network, X, cfg, false, rng=rng)
    acc = randindex(Y, final_labels)
    Logging.info(@sprintf("S: %d %d %s %s %0.4f %0.4f %0.4f %0.4f",
                          train_epochs, seed, problem, fname, acc[1], acc[2],
                          acc[3], acc[4]))

    final_labels
end
