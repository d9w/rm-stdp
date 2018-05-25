struct Network
    neurons::Array{Float64}
    nin::Int64
    nout::Int64
    connections::BitArray
    weights::Array{Float64}
    ge::Array{Float64}
    gi::Array{Float64}
    trace::Array{Float64}
    exc::BitArray
    inh::BitArray
    inputs::BitArray
    outputs::BitArray
    da::Array{Float64}
    history::Dict
    cfg::Dict
end

function Network(n_neurons::Int64, n_input::Int64, n_output::Int64,
                 connectivity::Float64, cfg::Dict;
                 n_exc::Int64 = Int64(round(0.8 * n_neurons)))
    # starting membrane threshold for all neurons
    neurons = zeros(n_neurons, 3)
    neurons[:, 1] = cfg["vreset"]
    neurons[:, 2] = 0.0 # homeostasis
    neurons[:, 3] = 0.0 # refractory
    # index of exc neurons
    n_inh = n_neurons - n_exc
    exc = [trues(n_exc); falses(n_inh)]
    shuffle!(exc)
    inh = .~(exc)
    # set weights
    weights = cfg["wstart"] * ones(n_neurons, n_neurons)
    weights[.~(exc), :] = cfg["winh"]
    # weights .*= (1.0 - eye(n_neurons, n_neurons))
    connections = (rand(n_neurons, n_neurons) .< connectivity)
    weights .*= connections
    # set input neurons
    inputs = falses(n_neurons)
    inds = shuffle(find(exc))
    inputs[inds[1:n_input]] = true
    # set output neurons
    outputs = falses(n_neurons)
    inds = shuffle(find(exc .& .~(inputs)))
    outputs[inds[1:n_output]] = true
    trace = zeros(n_neurons, n_neurons)
    ge = zeros(n_neurons, n_neurons)
    gi = zeros(n_neurons, n_neurons)
    da = [0.0]
    history = Dict()
    history["spikes"] = BitArray(n_neurons, 0)
    Network(neurons, n_input, n_output, connections, weights, ge, gi, trace,
            exc, inh, inputs, outputs, da, history, cfg)
end

function input!(n::Network, input_spikes::BitArray)
    inds = find(n.inputs)[find(input_spikes)]
    n.neurons[inds, 1] .+= n.cfg["vinput"]
    nothing
end

function spike!(n::Network)
    # calculate spikes, reset spiked pre-synaptic membranes
    spikes = n.neurons[:, 1] .>= (n.cfg["vthresh"] .+ n.neurons[:, 2])
    spikes[n.exc] .&= (n.neurons[n.exc, 3] .> n.cfg["refrac_e"])
    spikes[n.inh] .&= (n.neurons[n.inh, 3] .> n.cfg["refrac_i"])
    # calculate conductance
    espikes = spikes .& n.exc
    n.ge[espikes, :] .+= n.weights[espikes, :]
    ispikes = spikes .& n.inh
    n.gi[ispikes, :] .+= n.weights[ispikes, :]
    # calculate post-synaptic membrane delta
    v = n.neurons[:, 1]
    dv = ((n.cfg["eleak"] - v) + (sum(n.ge, 1)' .* (n.cfg["eexc"] - v))
          + (sum(n.gi, 1)' .* (n.cfg["einh"] - v))
          + n.cfg["noise"] * rand(size(v)))
    dv[n.exc] /= n.cfg["tme"]
    dv[n.inh] /= n.cfg["tmi"]
    # update neuron state values
    nspikes = .~(spikes)
    n.neurons[spikes, 1] .= n.cfg["vreset"]
    n.neurons[nspikes, 1] .+= dv[nspikes]
    n.neurons[spikes, 2] .+= n.cfg["theta_plus"]
    n.neurons[nspikes, 2] .-= n.neurons[nspikes, 2] / n.cfg["ttheta"]
    n.neurons[spikes, 3] .= 0.0
    n.neurons[nspikes, 3] .+= 1.0
    n.ge .-= n.ge / n.cfg["te"]
    n.gi .-= n.gi / n.cfg["ti"]
    # n.history["spikes"] = [n.history["spikes"] spikes]
    spikes
end

function learn!(n::Network, spikes::BitArray)
    n.trace[spikes, :] .+= 1
    n.trace .-= (n.trace / n.cfg["ttrace"])
    if n.da[1] > 0.0
        dw = (exp(-n.cfg["lr"]) * n.da[1] * (n.trace[n.exc, spikes] - n.cfg["target"]) .*
              (n.cfg["wmax"] - n.weights[n.exc, spikes]).^n.cfg["mu"])
        n.weights[n.exc, spikes] .+= dw
        n.weights[n.weights .> n.cfg["wmax"]] .= n.cfg["wmax"]
        n.weights[n.weights .< 0.0] .= 0.0
        n.weights .*= n.connections
        n.da .-= n.da / n.cfg["tdop"]
    end
    nothing
end

function step!(n::Network, train::Bool=true)
    spikes = spike!(n)
    if train
        learn!(n, spikes)
    end
    spikes[n.outputs]
end

function step!(n::Network, input_spikes::BitArray, train::Bool=true)
    input!(n, input_spikes)
    step!(n, train)
end

