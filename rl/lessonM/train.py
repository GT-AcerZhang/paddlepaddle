import numpy as np
import os
from parl.utils import logger, ReplayMemory, action_mapping
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from utils import Config, plot_reward
from tqdm import tqdm

class Trainer(object):
    def __init__(self, config, env, cls, name):
        self.config = Config(config['main'])
        self.agentArgs = config[name]
        self.agentCls = cls
        self.name = name
            
    def run_game(self):

        config = self.config
        n = config.runs_per_agent
        prev_best_reward = -1000

        for run in range(n):

            # potentially, we can change the goals as agent picks up more skills
            env = eval(config.environment)
            test_env = eval(config.environment)
            cLoss, aLoss = [], []

            # 0. instantiate an agent instance of this class
            agent = self.agentCls(**self.agentArgs)
            obs_dim, act_dim = agent.obs_dim, agent.act_dim

            # 1. instantiate a memory pool and warm up
            rpm = ReplayMemory(config.memory_size, obs_dim, act_dim)

            # 2. set up logging file
            save_dir = config.log_path + "{}_{}".format(self.name, run+1)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 3. start training
            test_flag, total_steps = 0, 0
            train_rewards, test_means, test_stds = [], [], []
            pbar = tqdm(total=config.train_total_steps)
            while total_steps < config.train_total_steps:
                
                para = [config.reward_scale, config.warmup_size, config.batch_size, config.expl_noise]
                train_reward, steps, costC, costA = run_train_episode(env, agent, rpm, *para)
                
                total_steps += steps
                train_rewards.append(train_reward)
                cLoss.append(costC); aLoss.append(costA)

                pbar.set_description('Steps: {} Reward: {}'.format(total_steps, train_reward))
                pbar.update(steps)
                
                # 4. start testing
                if total_steps // config.test_every_steps >= test_flag:
                    while total_steps // config.test_every_steps >= test_flag:
                        test_flag += 1
                    r_mean, r_std = run_evaluate_episode(test_env, agent)
                    logger.info('Steps {}, Evaluate reward: {}'.format(total_steps, r_mean))
                    test_means.append(r_mean)
                    test_stds.append(r_std)
                    if config.save_model and r_mean > prev_best_reward:
                        prev_best_reward = r_mean
                        ckpt = save_dir + '/Steps_{}_reward_{}.ckpt'.format(total_steps, int(r_mean))
                        agent.save(ckpt, program=agent.pred_program)
                    np.savez(save_dir+'/record.npz', train=train_rewards, mean=test_means, std=test_stds, closs=cLoss, aloss=aLoss)
            if config.visual_result:
                plot_reward(train_rewards)
                plot_reward(test_means, test_stds)
            
        

def run_train_episode(env, agent, rpm, reward_scale, warmup_size, batch_size, expl_noise):
    obs = env.reset()
    total_reward, steps = 0, 0
    critic_cost, actor_cost = 0, 0
    low_act, high_act = env.action_space.low[0], env.action_space.high[0]
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        if rpm.size() < warmup_size:
            action = env.action_space.sample()
        elif hasattr(agent, "sample_program"):
            action = agent.sample(batch_obs.astype('float32'))
        else:
            action = agent.predict(batch_obs.astype('float32'))
            action = np.clip(np.random.normal(action, high_act*expl_noise), -high_act, high_act)
        action = np.clip(action/high_act, -1.0, 1.0)
        action = action_mapping(action, low_act, high_act)
        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, reward_scale * reward, next_obs, done)

        if rpm.size() > warmup_size:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(batch_size)
            critic_cost, actor_cost = agent.learn(batch_obs, batch_action, batch_reward,batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps, critic_cost, actor_cost

# def write2txt(record, dir, name):
#     # to load this file
#     # with open('records.txt') as f:   
#     #      records = eval(f.read())
#     with open(dir + '{}.txt'.format(name), 'w') as f:
#         f.write(repr(record))

def run_evaluate_episode(env, agent, render=False):
    eval_reward = []
    low_act, high_act = env.action_space.low[0], env.action_space.high[0]
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.clip(action/high_act, -1.0, 1.0)
            action = action_mapping(action, low_act, high_act)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            total_reward += reward
            steps += 1
            if render:
                env.render()
            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward), np.std(eval_reward)