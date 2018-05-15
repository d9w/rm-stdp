using ArgParse
include("clustering.jl")
include("datasets.jl")

settings = ArgParseSettings()
@add_arg_table settings begin
    "--problem"
    arg_type = String
    default = "iris"
    "--fname"
    arg_type = String
    default = "lif"
    "--seed"
    arg_type = Int
    default = 0
    "--logfile"
    arg_type = String
    default = "stdp.log"
    "--train_epochs"
    arg_type = Int
    default = 1
    "--weight_mean"
    arg_type = Float64
    default = 0.5
    "--weight_std"
    arg_type = Float64
    default = 0.1
    "--t_train"
    arg_type = Int
    default = 350
    "--t_blank"
    arg_type = Int
    default = 150
    "--fr"
    arg_type = Float64
    default = 65.0
    "--pre_target"
    arg_type = Float64
    default = 0.4
    "--stdp_lr"
    arg_type = Float64
    default = 0.0001
    "--stdp_mu"
    arg_type = Float64
    default = 2.0
    "--inhib_weight"
    arg_type = Float64
    default = 0.1
    "--ika"
    arg_type = Float64
    default = 0.02
    "--ikb"
    arg_type = Float64
    default = 0.2
    "--ikd"
    arg_type = Float64
    default = 2.0
end
settings

args = parse_args(settings)
Logging.configure(filename=args["logfile"], level=INFO)

X, Y = get_data(args["problem"])
n_cluster = length(unique(Y))

stdp_labels = stdp_cluster(
    X, Y, n_cluster, x->x; seed=args["seed"], logfile=args["logfile"],
    problem=args["problem"], fname=args["fname"],
    train_epochs=args["train_epochs"], weight_mean=args["weight_mean"],
    weight_std=args["weight_std"], t_train=args["t_train"],
    t_blank=args["t_blank"], fr=args["fr"], pre_target=args["pre_target"],
    stdp_lr=args["stdp_lr"], stdp_mu=args["stdp_mu"],
    inhib_weight=args["inhib_weight"], ika=args["ika"],
    ikb=args["ikb"], ikd=args["ikd"])

acc = randindex(stdp_labels, Y)

Logging.info(@sprintf("E%0.6f", -acc[1]))
