using CGP
using Logging
using ArgParse

include("../graph_utils.jl")

function read_data(dfile::String="data/lif.data")
    df = open(dfile, "r")
    data = readdlm(dfile, ' ')
    ntrain = Int64(round(size(data, 1) * 3.0 / 4.0))
    ntest = size(data, 1) - ntrain
    training = data[1:ntrain, :]
    test = data[ntrain+(1:ntest), :]
    5, 5, training', test'
end

function regression(c::Chromosome, data::Array{Float64}, nin::Int64, nout::Int64)
    error = 0
    nsamples = size(data, 2)
    for d in 1:nsamples
        outputs = process(c, data[1:nin, d])
        for p in eachindex(outputs)
            error += (outputs[p] - data[nin+p, d])^2
        end
    end
    error /= nsamples
    -error
end

s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
    arg_type = Int
    default = 0
    "--log"
    arg_type = String
    required = true
end

args = parse_args(s)

CGP.Config.init("cfg/base.yaml")
CGP.Config.init("cfg/functions.yaml")

srand(args["seed"])
Logging.configure(filename=args["log"], level=INFO)
nin, nout, train, test = read_data()

fit = x->regression(x, train, nin, nout)
refit = x->regression(x, test, nin, nout)
maxfit, best = oneplus(CGPChromo, nin, nout, fit; seed=args["seed"],
                       record_best=true, record_fitness=refit)
