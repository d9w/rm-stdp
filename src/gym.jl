using PyCall

@pyimport gym
@pyimport pybullet_envs.bullet.simpleHumanoidGymEnv as humangym

function play_env(n::Network, env, tsim::Int64, n_actions::Int64)
    env[:seed](0)
    ob = env[:reset]()
    nin = sum(n.inputs)
    nout = sum(n.outputs)
    ob_low = env[:observation_space][:low]
    ob_high = env[:observation_space][:high]
    total_reward = 0.0
    done = false
    reward = 0.0

    while ~done
        nob = (ob - ob_low) ./ (ob_high - ob_low)
        append!(nob, [rand()])
        ins = rand(nin, tsim)
        inputs = falses(nin, tsim)
        for i in 1:nin
            inputs[i, :] = ins[i, :] .< nob[mod(i, length(nob))+1]
        end
        ocount = zeros(n_actions)
        for t in 1:tsim
            outs = step!(n, inputs[:, t])
            for o in 1:length(outs)
                ocount[mod(o, n_actions)+1] += 1
            end
        end
        action = indmax(ocount)
        ob, reward, done, _ = env[:step](action)
        total_reward += reward
    end

    total_reward
end

function repeat_trials(n::Network, env; tsim::Int64=10, ma_rate::Float64=0.9,
                       n_actions::Int64=2, n_trials::Int64=50)
    ma_reward = play_env(n, env, tsim, n_actions)

    for i in 1:n_trials
        reward = play_env(n, env, tsim, n_actions)
        dop = reward - ma_reward
        if dop > 0
            n.da[1] += dop
        end
        ma_reward = (1.0 - ma_rate) * ma_reward + ma_rate * reward
    end

    final_reward = mean(x->play_env(n, env, tsim, n_actions), 1:3)
end


