function lif(nstate::Array{Float64}, vstart::Float64, vthresh::Float64)
    outputs = zeros(5)
    if (nstate[1] >= vthresh) && nstate[2] < 0.8
        outputs[1] = vstart - nstate[1]
        outputs[2] = 1.0 - nstate[2]
        outputs[5] = 1.0
    else
        outputs[1] = -0.05 * nstate[1] + nstate[5]
        outputs[2] = -0.05 * nstate[2]
    end

    max.(-1.0, min.(1.0, outputs))
end

function izhikevich(nstate::Array{Float64}, vstart::Float64, vthresh::Float64,
                    vscale::Float64; a::Float64=0.02, b::Float64=0.2,
                    c::Float64=-65.0/vscale, d::Float64=2.0/vscale)
    v = nstate[1]
    u = nstate[3]
    I = nstate[5]

    fired = 0.0
    if nstate[1] >= vthresh
        v = c
        u = u + d
        fired = 1.0
    end

    c1 = 0.04 * vscale
    c2 = 140 / vscale
    dv = c1 * v^2 + 5 * v + c2 - u + I
    du = a * (b * v - u)

    outputs = [dv, 0.0, du, 0.0, fired]

    max.(-1.0, min.(1.0, outputs))
end
