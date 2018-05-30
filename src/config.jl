using YAML
using ArgParse

function add_args(s::ArgParseSettings)
    @add_arg_table(
        s,
        "--cfg", arg_type = String, default = "cfg/default.yaml",
        "--vinput", arg_type = Int64, default = 16,
        "--noise", arg_type = Float64, default = 5.0,
        "--theta_plus", arg_type = Float64, default = 0.4,
        "--refrac_e", arg_type = Float64, default = 15.0,
        "--refrac_i", arg_type = Float64, default = 4.0,
        "--ttheta", arg_type = Float64, default = 4.7,
        "--tdop", arg_type = Float64, default = 6.2,
        "--lr", arg_type = Float64, default = 6.0,
        "--target", arg_type = Float64, default = 0.4,
        "--wrandom", arg_type = Bool, default = true,
        "--wstart", arg_type = Float64, default = 0.9,
        "--winh", arg_type = Float64, default = 0.4,
        "--wmax", arg_type = Float64, default = 3.5,
        "--mu", arg_type = Float64, default = 0.1
    )
    s
end

function get_config(argtable::ArgParseSettings)
    argtable = add_args(argtable)
    args = parse_args(argtable)
    cfg = YAML.load_file(args["cfg"])

    for k in ["noise", "vinput", "theta_plus", "refrac_e", "refrac_i", "lr",
              "target", "wrandom", "wstart", "winh", "wmax", "mu"]
        cfg[k] = args[k]
    end

    for k in ["ttheta", "tdop"]
        cfg[k] = args[k] * 100
    end
    args, cfg
end
