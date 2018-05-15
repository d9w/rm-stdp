include("mnist.jl")

function get_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--seed"
        arg_type = Int
        default = 0
        "--log"
        arg_type = String
        required = true
        "--ea"
        arg_type = String
        required = true
        "--chromosome"
        arg_type = String
        required = true
        "--cfg"
        arg_type = String
        required = true
    end

    CGP.Config.add_arg_settings!(s)
end

args = parse_args(get_args())
CGP.Config.init(args["cfg"])
srand(args["seed"])
Logging.configure(filename=args["log"], level=INFO)
ea = eval(parse(args["ea"]))
ctype = eval(parse(args["chromosome"]))

maxfit, best = ea(ctype, 4, 2, stdp_mnist)
Logging.info(@sprintf("E%0.6f", -maxfit))
