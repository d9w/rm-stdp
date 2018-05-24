using YAML
using Logging
using ArgParse
using Gadfly

include("src/network.jl")

Gadfly.push_theme(Theme(major_label_font="Helvetica", major_label_font_size=28pt,
                        minor_label_font="Helvetica", minor_label_font_size=24pt,
                        key_title_font="Helvetica", key_title_font_size=16pt,
                        key_label_font="Helvetica", key_label_font_size=14pt,
                        line_width=0.8mm, point_size=0.5mm,
                        lowlight_color=c->RGBA{Float32}(c.r, c.g, c.b, 0.2),
                        default_color=colorant"#000000"))

colors = [colorant"#e41a1c", colorant"#377eb8", colorant"#4daf4a",
          colorant"#984ea3", colorant"#ff7f00", colorant"#ffff33"]

argtable = ArgParseSettings()
@add_arg_table argtable begin
    "--seed"
    arg_type = Int
    default = 0
    "--log"
    arg_type = String
    default = "timings.log"
    "--plot"
    arg_type = String
    default = "timings.pdf"
    "--cfg"
    arg_type = String
    default = "cfg/default.yaml"
    "--vinput"
    arg_type = Int64
    default = 15
    "--connectivity"
    arg_type = Float64
    default = 0.1
end
args = parse_args(argtable)

cfg = YAML.load_file(args["cfg"])
cfg["vinput"] = args["vinput"]

Logging.configure(filename=args["log"], level=INFO)

nn = 100
nin = 10
nout = 10
tsim = 1000

srand(args["seed"])
n = Network(nn, nin, nout, args["connectivity"], cfg)

inputs = rand(nin, tsim) .< (0.5 / nin)
spikes = falses(nn, tsim)
reward = 0

# plt = plot(x=1:nn, y=1:nn, color=n.weights, Geom.rectbin,
#            Guide.xlabel(nothing), Guide.ylabel(nothing))
# draw(PDF("weights_0.pdf", 8inch, 6inch), plt)

for t in 1:tsim
    input!(n, inputs[:, t])
    s = spike!(n)
    learn!(n, s)
    spikes[:, t] = s
    if inputs[1, t]
        reward = t + rand(1:10)
    end
    if t == reward
        n.da[1] += 1.0
    end
    if ((t > 0) && (mod(t, 10) == 0))
        Logging.info(@sprintf("E: %d %d %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f",
                              t, sum(spikes[:, t+(-9:0)]),
                              mean(n.ge), mean(n.gi), mean(n.neurons[:, 1]),
                              mean(n.neurons[:, 2]), mean(n.neurons[:, 3]),
                              mean(n.weights)))
    end
end

# plt = plot(x=1:nn, y=1:nn, color=n.weights, Geom.rectbin,
#            Guide.xlabel(nothing), Guide.ylabel(nothing))
# draw(PDF(string("weights_", tsim, ".pdf"), 8inch, 6inch), plt)

s = map(x->ind2sub(spikes, x), find(spikes))
plt = plot(x=map(x->x[2], s), y=map(x->x[1], s), Geom.point,
           Guide.xlabel(nothing), Guide.ylabel(nothing))

draw(PDF(args["plot"], 8inch, 6inch), plt)


