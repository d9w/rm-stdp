using JSON
using Logging
using ArgParse

include("src/config.jl")
include("src/network.jl")
include("src/datasets.jl")
include("src/classify.jl")

argtable = ArgParseSettings()
@add_arg_table(
    argtable,
    "--seed", arg_type = Int, default = 0,
    "--log", arg_type = String, default = "iris.log",
    "--data", arg_type = String, default = "iris",
    "--supervised", arg_type = Bool, default = true,
    "--n_epochs", arg_type = Int64, default = 1,
    "--connectivity", arg_type = Float64, default = 0.3,
    "--tinput", arg_type = Int64, default = 4,
    "--toutput", arg_type = Int64, default = 4,
    "--trest", arg_type = Int64, default = 7,
    "--ma_rate", arg_type = Float64, default = 0.8,
    "--fr", arg_type = Float64, default = 0.5,
    "--n", arg_type = Int64, default = 12,
    "--nratio", arg_type = Float64, default = 2.8,
)

args, cfg = get_config(argtable)
srand(args["seed"])

Logging.configure(filename=args["log"], level=INFO)
Logging.info(@sprintf("Arguments: %s", JSON.json(args)))
Logging.info(@sprintf("Config: %s", JSON.json(cfg)))

# nin, nout, nn
X, Y = get_data(args["data"])
nin = size(X, 1) * args["n"]
nout = length(unique(Y)) * args["n"]
nn = Int64(round((nin + nout) * args["nratio"]))

n = Network(nn, nin, nout, args["connectivity"], cfg)

pcfg = Dict()
pcfg["n_classes"] = length(unique(Y))
for k in ["tinput", "toutput", "trest"]
    pcfg[k] = args[k] * 10
end
for k in ["ma_rate", "fr", "n_epochs", "seed", "supervised", "data"]
    pcfg[k] = args[k]
end

fit = run_classify(n, X, Y, pcfg)
Logging.info(@sprintf("E%0.6f", -fit))
