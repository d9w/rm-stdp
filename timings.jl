using YAML
using Logging
using ArgParse

include("src/network.jl")
include("src/plot.jl")

argtable = ArgParseSettings()
@add_arg_table argtable begin
    "--seed"
    arg_type = Int
    default = 0
    "--log"
    arg_type = String
    default = "timings.log"
    "--plot"
    arg_type = String
    default = "timings.pdf"
    "--cfg"
    arg_type = String
    default = "cfg/default.yaml"
    "--vinput"
    arg_type = Int64
    default = 15
    "--connectivity"
    arg_type = Float64
    default = 0.1
end
args = parse_args(argtable)

cfg = YAML.load_file(args["cfg"])
cfg["vinput"] = args["vinput"]

Logging.configure(filename=args["log"], level=INFO)

nn = 100
nin = 10
nout = 10
tsim = 1000

srand(args["seed"])
n = Network(nn, nin, nout, args["connectivity"], cfg)

inputs = rand(nin, tsim) .< (0.5 / nin)
reward = 0

for t in 1:tsim
    input!(n, inputs[:, t])
    s = spike!(n)
    learn!(n, s)
    if inputs[1, t] && inputs[4, t] && inputs[7, t]
        reward = t + rand(1:10)
    end
    if t == reward
        n.da[1] += 1.0
    end
    if ((t > 0) && (mod(t, 10) == 0))
        Logging.info(@sprintf("E: %d %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f", t,
                              mean(n.ge), mean(n.gi), mean(n.neurons[:, 1]),
                              mean(n.neurons[:, 2]), mean(n.neurons[:, 3]),
                              mean(n.weights)))
    end
end

plot_spikes(n, args["plot"])
