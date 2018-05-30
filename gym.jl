using JSON
using Logging
using ArgParse

include("src/config.jl")
include("src/network.jl")
include("src/gym.jl")

argtable = ArgParseSettings()
@add_arg_table(
    argtable,
    "--seed", arg_type = Int, default = 0,
    "--log", arg_type = String, default = "gym.log",
    "--id", arg_type = String, default = "CartPole-v0",
    "--plot", arg_type = String, default = "",
    "--n_trials", arg_type = Int64, default = 1,
    "--connectivity", arg_type = Float64, default = 0.3,
    "--tinput", arg_type = Int64, default = 4,
    "--toutput", arg_type = Int64, default = 4,
    "--trest", arg_type = Int64, default = 7,
    "--ma_rate", arg_type = Float64, default = 0.8,
    "--fr", arg_type = Float64, default = 0.5,
    "--n", arg_type = Int64, default = 12,
    "--nratio", arg_type = Float64, default = 2.8,
    "--onpolicy", arg_type = Bool, default = true,
    "--dmethod", arg_type = Int64, default = 1
)

args, cfg = get_config(argtable)

Logging.configure(filename=args["log"], level=INFO)
Logging.info(@sprintf("Arguments: %s", JSON.json(args)))
Logging.info(@sprintf("Config: %s", JSON.json(cfg)))

env = gym.make(args["id"])
nin = env[:observation_space][:shape][1] * args["n"]
naction = env[:action_space][:n]
nout = naction * args["n"]
nn = Int64(round((nin + nout) * args["nratio"]))

srand(args["seed"])
n = Network(nn, nin, nout, args["connectivity"], cfg)

pcfg = Dict()
pcfg["n_actions"] = naction
for k in ["tinput", "toutput", "trest"]
    pcfg[k] = args[k] * 10
end
for k in ["ma_rate", "fr", "n_trials", "seed", "id", "onpolicy", "dmethod"]
    pcfg[k] = args[k]
end

fit = repeat_trials(n, env, pcfg)
Logging.info(@sprintf("E%0.6f", -fit))

if length(args["plot"]) > 0
    include("src/plot.jl")
    plot_spikes(n, args["plot"])
end
