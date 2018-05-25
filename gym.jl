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
    "--id"
    arg_type = String
    default = "HalfCheetahBulletEnv-v0"
    "--plot"
    arg_type = String
    default = ""
    "--cfg"
    arg_type = String
    default = "cfg/default.yaml"
    "--n_trials"
    arg_type = Int64
    default = 1
    "--connectivity"
    arg_type = Float64
    default = 0.1
    "--tsim"
    arg_type = Int64
    default = 6
    "--trest"
    arg_type = Int64
    default = 2
    "--ma_rate"
    arg_type = Float64
    default = 0.4
    "--vinput"
    arg_type = Int64
    default = 10
    "--noise"
    arg_type = Float64
    default = 6.0
    "--theta_plus"
    arg_type = Float64
    default = 0.1
    "--refrac_e"
    arg_type = Float64
    default = 6.0
    "--refrac_i"
    arg_type = Float64
    default = 11.0
    "--ttheta"
    arg_type = Float64
    default = 780.0
    "--tdop"
    arg_type = Float64
    default = 320.0
    "--lr"
    arg_type = Float64
    default = 7.0
    "--target"
    arg_type = Float64
    default = 0.9
    "--wrandom"
    arg_type = Bool
    default = false
    "--wstart"
    arg_type = Float64
    default = 0.9
    "--winh"
    arg_type = Float64
    default = 0.5
    "--wmax"
    arg_type = Float64
    default = 2.1
    "--mu"
    arg_type = Float64
    default = 0.4
    "--fr"
    arg_type = Float64
    default = 0.6
    "--n"
    arg_type = Int64
    default = 10
    "--nratio"
    arg_type = Float64
    default = 2.5
    "--outrate"
    arg_type = Float64
    default = 0.4
    "--rmethod"
    arg_type = String
    default = "episodic"
end
args = parse_args(argtable)

cfg = YAML.load_file(args["cfg"])

for k in ["noise", "vinput", "theta_plus", "refrac_e", "refrac_i", "lr",
          "target", "wrandom", "wstart", "winh", "wmax", "mu"]
    cfg[k] = args[k]
end

for k in ["ttheta", "tdop"]
    cfg[k] = args[k] * 10
end

Logging.configure(filename=args["log"], level=INFO)
Logging.info(repr(cfg))

env = gym.make(args["id"])
nin = env[:observation_space][:shape][1] * args["n"]
naction = env[:action_space][:shape][1]
nout = naction * 2 * args["n"]
nn = Int64(round((nin + nout) * args["nratio"]))

srand(args["seed"])
n = Network(nn, nin, nout, args["connectivity"], cfg)

pcfg = Dict()
pcfg["env"] = args["id"]
pcfg["n_actions"] = naction
pcfg["tsim"] = args["tsim"] * 10
pcfg["trest"] = args["trest"] * 10
for k in ["ma_rate", "fr", "outrate", "rmethod"]
    pcfg[k] = args[k]
end

if args["n_trials"] > 1
    fit = repeat_trials(n, env, pcfg, args["n_trials"])
else
    fit = play_env(n, env, pcfg, 0, true)
end
Logging.info(@sprintf("E%0.6f", -fit))

if length(args["plot"]) > 0
    include("src/plot.jl")
    plot_spikes(n, args["plot"])
end
