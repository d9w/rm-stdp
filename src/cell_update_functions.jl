using CGP
using Logging
using ArgParse

pmin = -100
pmax = 100

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

function izhikevich(u::Float64, v::Float64, I::Float64; a::Float64=0.02,
                    b::Float64=0.2, c::Float64=-65.0, d::Float64=2.0)
    if v >= 30
        v = c
        u = u + d
    end

    v = v+0.5*(0.04*v^2+5*v+140-u+I)
    v = v+0.5*(0.04*v^2+5*v+140-u+I)
    u = u+a*(b*v-u)

    v, u
end

function get_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--seed"
        arg_type = Int
        default = 0
        "--log"
        arg_type = String
        required = true
        "--function"
        arg_type = String
        default = "izhikevich"
        "--cfg"
        arg_type = String
        default = "cfg/base.yaml"
    end

    CGP.Config.add_arg_settings!(s)
end

args = parse_args(get_args())
println(args)
CGP.Config.init(Dict([k=>args[k] for k in setdiff(
    keys(args), ["seed", "log", "function", "cfg"])]...))

CGP.Config.init("cfg/base.yaml")
CGP.Config.init("cfg/classic.yaml")
# CGP.Config.init(args["cfg"])

srand(args["seed"])
Logging.configure(filename=args["log"], level=INFO)
ea = oneplus
ctype = CGP.PCGPChromo

nin = 3
nout = 2
# f = eval(parse(args["function"]))
fit = x->fmatch(x, izhikevich, nin, nout)
maxfit, best = ea(ctype, nin, nout, fit; seed=args["seed"])

best_ind = ctype(best, nin, nout)
test_fit = fitness(best_ind, test, nin, nout)
Logging.info(@sprintf("T: %d %0.8f %0.8f %d %d %s %s",
                      args["seed"], maxfit, test_fit,
                      sum([n.active for n in best_ind.nodes]),
                      length(best_ind.nodes), args["ea"], args["chromosome"]))

Logging.info(@sprintf("E%0.8f", -maxfit))
