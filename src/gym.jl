using PyCall

@pyimport gym

function play_env(n::Network, env, pcfg::Dict, seed::Int64, trial::Int64,
                  train::Bool=true)
    env[:seed](seed)
    ob = env[:reset]()
    ehigh = min.(env[:observation_space][:high], 20)
    elow = max.(env[:observation_space][:low], -20)
    ma_reward = -Inf
    max_reward = 0.0
    reward = -Inf
    done = false
    step = 0
    bad_step = 0

    while ~done
        nob = min.(max.(((ob - elow) ./ (ehigh - elow)), 0.0), 1.0) .* pcfg["fr"]
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
        ob, r, done, _ = env[:step](indmax(ocount)-1)
        reward = r
        if pcfg["id"] == "Acrobot-v1"
            reward = acos(ob[1]) # max angle
            max_reward = max(max_reward, reward)
        elseif pcfg["id"] == "CartPole-v0"
            reward = 1.0 - acos(cos(ob[3])) # 1 - angle
            max_reward += 1.0
        elseif pcfg["id"] == "MountainCar-v0"
            reward = (ob[1] - elow[1]) / (ehigh[1] - elow[1])  # mountain car
            max_reward = max(max_reward, reward)
        else
            max_reward += reward
        end
        if ma_reward == -Inf
            ma_reward = reward
        else
            ma_reward = (1.0 - pcfg["ma_rate"]) * ma_reward + pcfg["ma_rate"] * reward
        end
        dop = 0.0
        if ma_reward != 0.0
            dop = reward / ma_reward
        end
        if pcfg["onpolicy"]
            if pcfg["dmethod"] == 1
                if dop > 1.0
                    n.da[1] += (dop - 1.0)
                end
            elseif pcfg["dmethod"] == 2
                n.da[1] *= dop
            elseif pcfg["dmethod"] == 3
                n.da[1] = dop
            end
        end
        step += 1
        Logging.info(@sprintf("S: %s %d %d %d %d %0.6f %0.6f %0.6f %0.6f",
                              pcfg["id"], trial, step, maxcount, mincount,
                              max_reward, reward, ma_reward, dop))
        if bad_step > 20
            return -1e4
        end
    end

    max_reward
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
        if dop > 1.0 && ~pcfg["onpolicy"]
            n.da[1] += (dop - 1.0)
        end
        ma_reward = (1.0 - pcfg["ma_rate"]) * ma_reward + pcfg["ma_rate"] * reward
        Logging.info(@sprintf("T: %s %d %d %0.6f %0.6f %0.6f %e %e",
                              pcfg["id"], i, seed, reward, ma_reward, dop,
                              sum(abs.(n.weights - init_weights)) / length(n.weights),
                              sum(abs.(n.weights - weights)) / length(n.weights)))
        seed += 1
    end

    mean(x->play_env(n, env, pcfg, pcfg["seed"]+x, pcfg["n_trials"]+x, false), 1:3)
end
