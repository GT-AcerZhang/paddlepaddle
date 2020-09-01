
import argparse
import parl
from parl import layers
import paddle.fluid as fluid
import numpy as np
import os
import gym
from parl.utils import logger

from model import Model_v0, Model_v1, Model_v2
from parl.algorithms import DQN
from agent import Agent 
from replay_memory import ReplayMemory

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等

# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done)) 
        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)  # s,a,r,s',done 
        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def choose_model(version, **kwargs):
    return eval('Model_%s' %version)(**kwargs)

def main():
    env = gym.make(
        'MountainCar-v0'
    )  # expect reward > -110?
    action_dim = env.action_space.n  # MountainCar-v0: 3
    obs_shape = env.observation_space.shape  # MountainCar-v0: (2,)
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = choose_model(args.model, act_dim=action_dim)
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=args.lr)
    agent = Agent(algorithm, obs_dim=obs_shape[0], act_dim=action_dim,
        e_greed=0.5,             # 有一定概率随机选取动作，探索
        e_greed_decrement=4e-5)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)
    max_episode = 1500

    # start train
    episode = 0
    while episode < max_episode:  
        # train part
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            episode += 1

        # test part
        eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
        logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
            episode, agent.e_greed, eval_reward))

    # 训练结束，保存模型
    save_path = 'saved_model/dqn_model_%s_%s.ckpt' %(args.model, args.lr)
    agent.save(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lr', default=3e-4, type=float, help='learning_rate')
    parser.add_argument('-m', dest='model', default='v1', help='neural_network')
    args = parser.parse_args()
    print("Start Training: learning rate = %.4f, model = model %s" %(args.lr, args.model))
    print("#"*50)
    logger.set_dir(os.path.join('./train_log', "%s_%s"%(args.model, args.lr)))
    main()