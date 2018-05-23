struct Network
    neurons::Array{Float64}
    weights::Array{Float64}
    ge::Array{Float64}
    gi::Array{Float64}
    trace::Array{Float64}
    excitatory::BitArray
    inputs::BitArray
    outputs::BitArray
    da::Array{Float64}
    cfg::Dict
    rng::MersenneTwister
end

function Network(n_neurons::Int64, n_input::Int64, n_output::Int64,
                 cfg::Dict; rng::MersenneTwister=MersenneTwister(0),
                 n_exc::Int64 = Int64(round(0.8 * n_neurons)))
    # starting membrane threshold for all neurons
    neurons = zeros(n_neurons, 1)
    neurons[:, 1] = cfg["vreset"]
    # index of excitatory neurons
    n_inh = n_neurons - n_exc
    excitatory = [trues(n_exc); falses(n_inh)]
    shuffle!(excitatory)
    # set weights
    weights = rand(n_neurons, n_neurons)
    weights[.~(excitatory), :] = 1.0
    weights .*= (1.0 - eye(n_neurons, n_neurons))
    weights .*= (rand(rng, n_neurons, n_neurons) .< cfg["connectivity"])
    # set input neurons
    inputs = falses(n_neurons)
    inds = shuffle(find(excitatory))
    inputs[inds[1:n_input]] = true
    # set output neurons
    outputs = falses(n_neurons)
    inds = shuffle(find(excitatory .& .~(inputs)))
    outputs[inds[1:n_output]] = true
    trace = zeros(n_neurons, n_neurons)
    ge = zeros(n_neurons, n_neurons)
    gi = zeros(n_neurons, n_neurons)
    da = [0.0]
    Network(neurons, weights, ge, gi, trace, excitatory, inputs, outputs,
            da, cfg, rng)
end

function spike!(n::Network, input_spikes::BitArray)
    # problem input
    inds = find(n.inputs)[find(input_spikes)]
    n.neurons[inds, 1] .+= n.cfg["vinput"]
    # calculate spikes, reset spiked pre-synaptic membranes
    spikes = n.neurons[:, 1] .>= n.cfg["vthresh"]
    n.neurons[spikes, 1] .= n.cfg["vreset"]
    # calculate conductance
    espikes = spikes .& n.excitatory
    n.ge[espikes, :] .+= n.weights[espikes, :]
    ispikes = spikes .& .~(n.excitatory)
    n.gi[ispikes, :] .+= n.weights[ispikes, :]
    # apply post-synaptic membrane delta
    v = n.neurons[:, 1]
    dv = ((n.cfg["eleak"] - v) + (sum(n.ge, 1)' .* (n.cfg["eexc"] - v))
          + (sum(n.gi, 1)' .* (n.cfg["einh"] - v))
          + n.cfg["thalamic"] * randn(n.rng, size(v))) / n.cfg["tm"]
    n.neurons[:, 1] .+= dv[:]
    n.ge .-= n.ge / n.cfg["te"]
    n.gi .-= n.gi / n.cfg["ti"]
    spikes
end

function learn!(n::Network, spikes::BitArray)
    n.trace[spikes, :] .+= 1
    dw = (n.cfg["lr"] * n.da[1] * (n.trace[:, spikes] - n.cfg["target"]) .*
          (n.cfg["wmax"] - n.weights[:, spikes]).^n.cfg["mu"])
    n.weights[:, spikes] .+= dw
    n.trace .-= (n.trace / n.cfg["tt"])
    n.da .*= (1.0 - n.cfg["dopamine_absorption"])
    nothing
end

function step!(n::Network, input_spikes::BitArray, train::Bool)
    spikes = spike!(n, input_spikes)
    if train
        learn!(n, spikes)
    end
    spikes[n.outputs]
end
