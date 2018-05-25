using YAML
using Logging
using ArgParse

include("src/network.jl")
include("src/gym.jl")

argtable = ArgParseSettings()
@add_arg_table argtable begin
    "--seed"
    arg_type = Int
    default = 0
    "--log"
    arg_type = String
    default = "gym.log"
    "--plot"
    arg_type = String
    default = ""
    "--cfg"
    arg_type = String
    default = "cfg/default.yaml"
    "--ntrials"
    arg_type = Int64
    default = 10
    "--connectivity"
    arg_type = Float64
    default = 0.1
    "--tsim"
    arg_type = Int64
    default = 10
    "--ma_rate"
    arg_type = Float64
    default = 0.1
    "--noise"
    arg_type = Float64
    default = 10.0
    "--vinput"
    arg_type = Int64
    default = 15
    "--theta_plus"
    arg_type = Float64
    default = 3.0
    "--refrac_e"
    arg_type = Float64
    default = 10.0
    "--refrac_i"
    arg_type = Float64
    default = 5.0
    "--ttheta"
    arg_type = Float64
    default = 100.0
    "--tdop"
    arg_type = Float64
    default = 20.0
    "--lr"
    arg_type = Float64
    default = 3.0
    "--target"
    arg_type = Float64
    default = 0.1
    "--wstart"
    arg_type = Float64
    default = 0.9
    "--winh"
    arg_type = Float64
    default = 1.0
    "--wmax"
    arg_type = Float64
    default = 2.0
    "--fr"
    arg_type = Float64
    default = 1.0
    "--n"
    arg_type = Int64
    default = 10
    "--nratio"
    arg_type = Float64
    default = 2.5
end
args = parse_args(argtable)

cfg = YAML.load_file(args["cfg"])

for k in ["noise", "vinput", "theta_plus", "refrac_e", "refrac_i", "lr",
          "target", "wstart", "winh", "wmax"]
    cfg[k] = args[k]
end

for k in ["ttheta", "tdop"]
    cfg[k] = args[k] * 10
end

Logging.configure(filename=args["log"], level=INFO)

env = gym.make("CartPole-v0")
nin = 4 * args["n"]
nout = 2 * args["n"]
nn = Int64(round((nin + nout) * args["nratio"]))

srand(args["seed"])
n = Network(nn, nin, nout, args["connectivity"], cfg)
fit = repeat_trials(n, env; tsim=args["tsim"]*10, ma_rate=args["ma_rate"],
                    n_trials=args["ntrials"], fr=args["fr"])
Logging.info(@sprintf("E%0.6f", -fit))

if length(args["plot"]) > 0
    include("src/plot.jl")
    plot_spikes(n, args["plot"])
end
