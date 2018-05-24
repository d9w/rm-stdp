using PyCall

@pyimport gym
@pyimport pybullet_envs.bullet.simpleHumanoidGymEnv as humangym

function play_env(n::Network, env, tsim::Int64, n_actions::Int64, fr::Float64,
                  train::Bool)
    # env[:seed](0)
    ob = env[:reset]()
    nin = sum(n.inputs)
    nout = sum(n.outputs)
    ob_low = env[:observation_space][:low]
    ob_high = env[:observation_space][:high]
    total_reward = 0.0
    done = false
    reward = 0.0
    good_step = 10
    bad_step = 0

    while ~done
        nob = (ob - ob_low) ./ (ob_high - ob_low)
        nob .*= fr
        ins = rand(nin, tsim)
        inputs = falses(nin, tsim)
        for i in 1:nin
            inputs[i, :] = ins[i, :] .< nob[mod(i, length(nob))+1]
        end
        ocount = zeros(n_actions)
        for t in 1:tsim
            outs = step!(n, inputs[:, t], train)
            for o in 1:length(outs)
                if outs[o]
                    ocount[mod(o, n_actions)+1] += 1
                end
            end
        end
        if std(ocount) == 0.0
            bad_step += 1
        else
            good_step += 1
        end
        if bad_step > good_step
            return 0
        end
        action = indmax(ocount) - 1
        ob, reward, done, _ = env[:step](action)
        total_reward += reward
    end

    total_reward
end

function repeat_trials(n::Network, env; tsim::Int64=10, ma_rate::Float64=0.9,
                       n_actions::Int64=2, n_trials::Int64=50, fr::Float64=1.0)
    ma_reward = play_env(n, env, tsim, n_actions, fr, true)

    init_weights = copy(n.weights)

    for i in 1:n_trials
        weights = copy(n.weights)
        reward = play_env(n, env, tsim, n_actions, fr, true)
        if reward == 0
            return 0
        end
        dop = 0.0
        if ma_reward > 0
            dop = reward / ma_reward
            if dop > 1.0
                n.da[1] += dop
            end
        end
        Logging.info(@sprintf("R: %d %0.6f %0.6f %0.6f %0.6f %0.6f",
                              i, reward, dop, ma_reward,
                              sum(abs.(n.weights - init_weights)) / length(n.weights),
                              sum(abs.(n.weights - weights)) / length(n.weights)))
        ma_reward = (1.0 - ma_rate) * ma_reward + ma_rate * reward
    end

    final_reward = mean(x->play_env(n, env, tsim, n_actions, fr, false), 1:10)
end


