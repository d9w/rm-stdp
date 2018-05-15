using ArgParse
using CGP
using YAML
include("clustering.jl")
include("datasets.jl")
include("seeds.jl")

settings = ArgParseSettings()
@add_arg_table settings begin
    "--seed"
    arg_type = Int
    default = 0
    "--expert"
    arg_type = String
    default = ""
    "--expfile"
    arg_type = String
    default = ""
    "--logfile"
    arg_type = String
    default = "stdp.log"
end

args = parse_args(settings)
srand(args["seed"])
Logging.configure(filename=args["logfile"], level=INFO)

CGP.Config.init("cfg/base.yaml")
CGP.Config.init("cfg/functions.yaml")
scfg = YAML.load_file("cfg/stdp.yaml")
fname = string(args["seed"])

function cluster_acc(c::Chromosome, X::Array{Float64}, Y::Array{Int64},
                     pname::String)
    n_cluster = length(unique(Y))
    nfunc = i->process(c, i)
    stdp_labels = stdp_cluster(
        X, Y, n_cluster, nfunc; seed=0, logfile=args["logfile"], problem=pname,
        fname=fname, train_epochs=scfg["train_epochs"],
        weight_mean=scfg["weight_mean"], weight_std=scfg["weight_std"],
        t_train=scfg["t_train"], t_blank=scfg["t_blank"], fr=scfg["fr"],
        pre_target=scfg["pre_target"], stdp_lr=scfg["stdp_lr"],
        stdp_mu=scfg["stdp_mu"], inhib_weight=scfg["inhib_weight"])
    acc = randindex(stdp_labels, Y)
    acc[1]
end

irisX, irisY = get_data("iris")

cluster_fit(c::Chromosome) = cluster_acc(c, irisX, irisY, "iris")

expert = nothing
if args["expert"] == "LIF"
    expert = to_chromo(lif_graph(-0.65, 0.3))
end
if args["expfile"] != ""
    experts = readdlm(args["expfile"], ',')
    s = mod(args["seed"], size(experts, 1))+1
    expert = PCGPChromo(experts[s, :], 5, 5)
end

maxfit, best = oneplus(PCGPChromo, 5, 5, cluster_fit; seed=args["seed"], expert=expert)

yeastX, yeastY = get_data("yeast")
yeast_acc = cluster_acc(best, yeastX, yeastY, "yeast")
Logging.info(@sprintf("F: %d %d %0.5f %0.5f %d %d",
seed, eval_count, maxfit, yeast_acc,
sum([n.active for n in best.nodes]),
length(best.nodes)))
