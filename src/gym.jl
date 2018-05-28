using PyCall

@pyimport gym

function play_env(n::Network, env, pcfg::Dict, seed::Int64, trial::Int64,
                  train::Bool=true)
    env[:seed](seed)
    ob = env[:reset]()
    ehigh = min.(env[:observation_space][:high], 20)
    elow = max.(env[:observation_space][:low], 20)
    total_reward = 0.0
    reward = 0.0
    done = false
    step = 0
    bad_step = 0

    while ~done
        nob = min.(max.(((ob - ehigh) / (ehigh - elow)), 0.0), 1.0) * pcfg["fr"]
        ins = rand(n.nin, pcfg["tinput"])
        inputs = falses(n.nin, pcfg["tinput"])
        for i in 1:n.nin
            inputs[i, :] = ins[i, :] .< nob[mod(i, length(nob))+1]
        end
        for t in 1:pcfg["tinput"]
            outs = step!(n, inputs[:, t], train)
        end
        ocount = zeros(pcfg["n_actions"])
        for t in 1:pcfg["toutput"]
            outs = step!(n, train)
            for o in 1:length(outs)
                # if outs[o]
                if rand() < 0.3
                    ocount[mod(o, length(ocount))+1] += 1.0
                end
            end
        end
        for t in 1:pcfg["trest"]
            step!(n, train)
        end
        maxcount = maximum(ocount)
        mincount = minimum(ocount)
        if maxcount == mincount
            bad_step += 1
        else
            bad_step = 0
        end
        ob, reward, done, _ = env[:step](indmax(ocount)-1)
        total_reward += reward
        step += 1
        if bad_step > 10
            return -1e4
        end
    end

    total_reward
end

function repeat_trials(n::Network, env, pcfg::Dict)
    init_weights = copy(n.weights)
    seed = 0
    ma_reward = play_env(n, env, pcfg, seed, 0, false)
    if ma_reward == -1e4
        return ma_reward
    end

    for i in 1:pcfg["n_trials"]
        weights = copy(n.weights)
        reward = play_env(n, env, pcfg, seed, i, true)
        if reward == -1e4
            return reward
        end
        dop = 0.0
        if ma_reward != 0.0
            dop = reward / ma_reward
        end
        if dop > 1.0
            n.da[1] += (dop - 1.0)
        end
        ma_reward = (1.0 - pcfg["ma_rate"]) * ma_reward + pcfg["ma_rate"] * reward
        Logging.info(@sprintf("T: %s %d %d %0.6f %0.6f %0.6f %e %e",
                              pcfg["env"], i, seed, reward, ma_reward, dop,
                              sum(abs.(n.weights - init_weights)) / length(n.weights),
                              sum(abs.(n.weights - weights)) / length(n.weights)))
        if ~pcfg["repeat"] || dop <= 1.0
            seed += 1
        end
    end

    mean(x->play_env(n, env, pcfg, pcfg["seed"]+x, pcfg["n_trials"]+x, false), 1:5)
end
