using PyCall

@pyimport gym
@pyimport pybullet_envs.bullet.simpleHumanoidGymEnv as humangym

function play_env(n::Network, env, pcfg::Dict, trial::Int64=1, train::Bool=true)
    ob = env[:reset]()
    scale = maximum(abs.(ob))
    total_reward = 0.0
    done = false
    reward = 0.0
    ma_reward = -Inf
    pcfg["outrate"] = 1.0
    step = 0

    init_weights = copy(n.weights)

    while ~done
        weights = copy(n.weights)
        # nob = ((ob - env[:observation_space][:low]) ./
        #        (env[:observation_space][:high] - env[:observation_space][:low]))
        nob = (ob + scale) / 2 * scale
        nob .*= pcfg["fr"]
        ins = rand(n.nin, pcfg["tsim"])
        inputs = falses(n.nin, pcfg["tsim"])
        for i in 1:n.nin
            inputs[i, :] = ins[i, :] .< nob[mod(i, length(nob))+1]
        end
        ocount = zeros(2 * pcfg["n_actions"])
        for t in 1:pcfg["tsim"]
            outs = step!(n, inputs[:, t], train)
            for o in 1:length(outs)
                if outs[o]
                    ocount[mod(o, length(ocount))+1] += 1.0
                end
            end
        end
        maxcount = maximum(ocount)
        mincount = minimum(ocount)
        action = zeros(pcfg["n_actions"])
        for i in 1:pcfg["n_actions"]
            ca = ocount[(i-1)*2 + 1]
            cb = ocount[(i-1)*2 + 2]
            if ca+cb > 0
                action[i] = (ca - cb) / (ca + cb)
            else
                action[i] = 0.0
            end
        end
        ob, reward, done, _ = env[:step](action)
        nscale = maximum(abs.(ob))
        if nscale > scale
            scale = nscale
        end
        total_reward += reward
        dop = 0.0
        if ma_reward == -Inf
            ma_reward = reward
        else
            if ma_reward > 0
                dop = reward / ma_reward
                if train && dop > 1.0 && pcfg["rmethod"] == "continuous"
                    n.da[1] += dop
                end
            end
            ma_reward = (1.0 - pcfg["ma_rate"]) * ma_reward + pcfg["ma_rate"] * reward
        end
        Logging.info(@sprintf("S: %s %d %d %d %d %0.6f %0.6f %0.6f %e %e",
                              pcfg["env"], trial, step, maxcount, mincount,
                              reward, ma_reward, dop,
                              sum(abs.(n.weights - init_weights)) / length(n.weights),
                              sum(abs.(n.weights - weights)) / length(n.weights)))
        step += 1
    end

    total_reward
end

function repeat_trials(n::Network, env, pcfg::Dict, n_trials=10);
    init_weights = copy(n.weights)
    ma_reward = play_env(n, env, pcfg, 0, false)
    p_reward = ma_reward
    change = 0

    for i in 1:n_trials
        weights = copy(n.weights)
        reward = play_env(n, env, pcfg, i, true)
        dop = 0.0
        if pcfg["rmethod"] == "episodic"
            if ma_reward > 0
                dop = reward / ma_reward
                if dop > 1.0
                    n.da[1] += dop
                end
            end
            ma_reward = (1.0 - pcfg["ma_rate"]) * ma_reward + pcfg["ma_rate"] * reward
        end
        dweights = sum(abs.(n.weights - weights)) / length(n.weights)
        if dweights > 0.0 && p_reward > reward
            change += 1.0
        end
        if i > 10 && i > 3 * (change + 1)
            break
        end
        p_reward = reward
        Logging.info(@sprintf("T: %s %d %0.6f %0.6f %0.6f %0.6f %e %e",
                              pcfg["env"], i, reward, ma_reward, dop, change,
                              sum(abs.(n.weights - init_weights)) / length(n.weights),
                              dweights))
        for t in 1:pcfg["trest"]
            spike!(n)
        end
    end

    map(x->play_env(n, env, pcfg, n_trials+x, false), 1:5)
end


