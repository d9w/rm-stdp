import MNIST
using CGP
using Logging
using ArgParse

global const nin = 784
global const nout = 10
global const N = 400
global const dt = 0.001
global const pre_dt = 20.0
global const pre_inc = 1.0
global const pre_target = 0.4
global const vstart = -65.0
global const vthresh = 30.0
global const vmin = -100.0
global const vmax = 50.0
global const wmax = 1.0
global const stdp_lr = 0.0001
global const stdp_mu = 2.0
global const inhib_weights = 0.1*(1.0-eye(N, N))

struct Network
    neurons::Array{Float64}
    weights::Array{Float64}
    trace::Array{Float64}
    classes::Array{Float64}
end

function process(inputs::Array{Float64})
    if inputs[4] == 1.0
        out = (vstart - vmin) / (vmax - vmin)
    else
        out = 0.95 * inputs[1] + inputs[3]
    end
    [out, 0.0]
    # rand(2)
end

function timestep(chromo::Chromosome, input_spikes::BitArray, network::Network,
                  train::Bool=false, label::Int64=0)
    inps = input_spikes' * network.weights
    fs = (network.neurons[:, 1] * (vmax - vmin) + vmin) .>= vthresh
    inps -= fs' * inhib_weights
    # TODO: check normalization
    inps = (inps'[:] .- vmin) ./ (vmax - vmin)
    class_labels = zeros(nout)
    for n in 1:N
        state = network.neurons[n, :]
        inputs = max.(-1.0, min.(1.0, [state; inps[n]; Float64(fs[n])]))
        # pout = CGP.process(chromo, inputs)
        pout = process(inputs)
        network.neurons[n, :] = pout
    end
    # stdp
    if train
        if label > 0
            network.classes[fs, label] .+= 1.0
        end
        network.trace[input_spikes, :] .+= pre_inc
        network.trace .-= (network.trace ./ pre_dt)
        network.weights[:, fs] .+= stdp_lr .* (network.trace[:, fs] .- pre_target) .* (
            (wmax .- network.weights[:, fs]).^stdp_mu)
        network.weights[network.weights .< 0] .= 0.0
    else
        fired = network.classes[fs, :]
        for i in 1:size(fired, 1)
            class_labels[indmax(fired[i, :])] += 1.0
        end
    end
    class_labels
end

function stdp_mnist(chromo::Chromosome)
    v = -65 .* ones(N)
    u = 0.2 .* v
    neurons = [v u]
    weights = rand(nin, N)
    trace = zeros(nin, N)
    class = zeros(N, nout)
    network = Network(neurons, weights, trace, class)
    acc_count = 0

    # training
    for img in 1:10
        fr = MNIST.trainfeatures(img) / 4.0
        label = Int64(1.0+MNIST.trainlabel(img))
        println(label)
        for t in 1:350
            input_spikes = rand(length(fr)) .< (dt * fr)
            timestep(chromo, input_spikes, network, true, label)
        end
        for t in 1:150
            input_spikes = BitArray(length(fr))
            timestep(chromo, input_spikes, network, true, 0)
        end
    end
    # testing
    for img in 1:100
        fr = MNIST.testfeatures(img) / 4.0
        label = Int64(1.0+MNIST.testlabel(img))
        class_labels = zeros(nout)
        for t in 1:350
            input_spikes = rand(length(fr)) .< (dt * fr)
            class_labels += timestep(chromo, input_spikes, network)
        end
        for t in 1:150
            input_spikes = BitArray(length(fr))
            class_labels += timestep(chromo, input_spikes, network)
        end
        println(label, " ", class_labels)
        # TODO: use a cluster accuracy metric
        if indmax(class_labels) == label
            acc_count += 1
        end
    end

    acc_count/100
end
