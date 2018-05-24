using Gadfly

Gadfly.push_theme(Theme(major_label_font="Helvetica", major_label_font_size=16pt,
                        minor_label_font="Helvetica", minor_label_font_size=14pt,
                        key_title_font="Helvetica", key_title_font_size=16pt,
                        key_label_font="Helvetica", key_label_font_size=14pt,
                        line_width=0.8mm, point_size=0.5mm,
                        lowlight_color=c->RGBA{Float32}(c.r, c.g, c.b, 0.2),
                        default_color=colorant"#000000"))

colors = [colorant"#e41a1c", colorant"#377eb8", colorant"#4daf4a",
          colorant"#984ea3", colorant"#ff7f00", colorant"#ffff33"]

function plot_spikes(n::Network, filename::String)
    spikes = n.history["spikes"]
    s = map(x->ind2sub(spikes, x), find(spikes))
    ntype = ["exc" for i in 1:length(n.exc)]
    ntype[n.inputs] = "input"
    ntype[n.outputs] = "output"
    ntype[n.inh] = "inh"
    y = map(x->x[1], s)
    color = map(x->ntype[x], y)
    plt = plot(x=map(x->x[2], s), y=y, color=color, Geom.point,
              Guide.xlabel(nothing), Guide.ylabel(nothing),
              Scale.color_discrete_manual(colors...))

    draw(PDF(filename, 8inch, 6inch), plt)
end
