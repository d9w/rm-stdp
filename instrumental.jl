using YAML
using Logging
using ArgParse

include("src/network.jl")

argtable = ArgParseSettings()
@add_arg_table argtable begin
    "--seed"
    arg_type = Int
    default = 0
    "--log"
    arg_type = String
    default = "instrumental.log"
    "--cfg"
    arg_type = String
    default = "cfg/default.yaml"
    "--vinput"
    arg_type = Int64
    default = 15
    "--connectivity"
    arg_type = Float64
    default = 0.1
    "--lr"
    arg_type = Float64
    default = 0.001
    "--target"
    arg_type = Float64
    default = 0.1
end
args = parse_args(argtable)

cfg = YAML.load_file(args["cfg"])
Logging.configure(filename=args["log"], level=INFO)

for k in ["vinput", "lr", "target"]
    cfg[k] = args[k]
end

nn = 1000
nin = 100
nout = 100
episodes = 5
tdelay = 20
treward = 100
trest = 1000

srand(args["seed"])
n = Network(nn, 2*nin, 2*nout, args["connectivity"], cfg)
ga = find(n.outputs)[1:nout]
gb = find(n.outputs)[nout+(1:nout)]

starta = mean(n.weights[n.inputs, ga])
startb = mean(n.weights[n.inputs, gb])

Logging.info(@sprintf("E: %d %d %d %d %d %0.7f %0.7f",
                      0, 0, 0, 0, 0,
                      mean(n.weights[n.inputs, ga]),
                      mean(n.weights[n.inputs, gb])))

for episode in 1:episodes
    nspikes = 0
    input!(n, [trues(nin); falses(nin)])
    nspikes += sum(spike!(n))
    ca = 0
    cb = 0
    for t in 1:tdelay
        spikes = spike!(n)
        nspikes += sum(spikes)
        outputs = spikes[n.outputs]
        ca += sum(outputs[1:nout])
        cb += sum(outputs[nout+(1:nout)])
    end
    delay = 0
    reward = 0
    if ca > cb
        reward = 0.1
        delay = Int64(round(treward * (cb / ca)))
    end
    for t in 1:delay
        nspikes += sum(spike!(n))
    end
    n.da[1] += reward
    tsim = trest - delay
    inputs = rand(2*nin, tsim) .< 0.1
    for t in 1:tsim
        input!(n, inputs[:, t])
        spikes = spike!(n)
        learn!(n, spikes)
        nspikes += sum(spikes)
    end
    Logging.info(@sprintf("E: %d %d %d %d %d %0.7f %0.7f",
                          episode, nspikes, ca, cb, delay,
                          mean(n.weights[n.inputs, ga]),
                          mean(n.weights[n.inputs, gb])))
end

enda = mean(n.weights[n.inputs, ga])
endb = mean(n.weights[n.inputs, gb])

fit = (enda - starta) - (endb - startb)
if (enda - starta) == 0.0
    fit = 0.0
end
Logging.info(@sprintf("E%0.6f", -fit))
