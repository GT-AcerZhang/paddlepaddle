#!/usr/bin/python

import argparse
import gym
import numpy as np
import time
import parl
from utils import *
from agent import Agent
from model import StockModel
from algorithm import DDPG
from parl.utils import logger, ReplayMemory
from env import StockTradingEnv
import pandas as pd
import os

# MAX_EPISODES = 5000
TAU = 0.005
MEMORY_SIZE = int(5e5)
ENV_SEED = 1
TRAIN_TOTAL_STEPS = 100000  # 总训练步数
TEST_EVERY_STEPS = 5000  # 每个N步评估一下算法效果，每次评估5个episode求平均reward

"""environment sketch
> shape: (5, 2)
> df.shape: (2335, 6)
> df.head()
Open        High         Low       Close   Adj Close       Volume                                          
2009-05-22  198.528534  199.524521  196.196198  196.946945  196.946945  3433700  
> prices.shape: (2310,)
> signal_features.shape: (2310, 2)
> max_possible_profit: 324533.2390176079

@properties:
    > env.trade_fee_bid_percent 1%
    > env.trade_fee_ask_percent 0.5%
    > env.frame_bound (30, 2335)

action:       buy=1 or sell=0
position:     short=0 or long=1
obs:          [(p_t, △p_t) x window_size=5]
"""

df = pd.read_csv('/home/aistudio/stock-anytrading/data/AAPL.csv')
df = df.sort_values('Date')

def main():

    env = StockTradingEnv(df)
    env.reset()
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    model = StockModel(act_dim=act_dim)
    algorithm = DDPG(model, gamma=dc, tau=TAU, actor_lr=al, critic_lr=cl)
    agent = Agent(algorithm, obs_dim, act_dim)

    total_steps, test_flag = 0, 0
    while total_steps < TRAIN_TOTAL_STEPS:
        # train part
        train_reward, steps, aloss, closs = run_episode(env, agent, rpm, bs)
        total_steps += steps
        logger.info("Step {}, Train Reward {}.".format(total_steps, train_reward))
        tb_logger.add_scalar(tag="Train/Reward", step=total_steps, value=train_reward)
        tb_logger.add_scalar(tag="Train/Actor", step=total_steps, value= aloss)
        tb_logger.add_scalar(tag="Train/Critic", step=total_steps, value= closs)

        # test part
        if total_steps // TEST_EVERY_STEPS >= test_flag:
            while total_steps // TEST_EVERY_STEPS >= test_flag:
                test_flag += 1            # keep increment until condition is violated
            eval_reward = evaluate(env, agent)
            logger.info('Step:{}, Test Reward:{}'.format(total_steps, eval_reward))
            tb_logger.add_scalar(tag="Test/Reward", step=total_steps, value=eval_reward)

            # 训练结束，保存模型
            save_path = 'check_point/ddpg_%s_%s_%s_%s.ckpt' %(bs,dc,al,cl)
            agent.save(save_path)

if __name__ == "__main__":
    # random action 
    # 'total_reward': 166.86534999999856
    # 'total_profit': 0.00013610198325005537
    # 'position': 0
    
    from visualdl import LogWriter
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, help='Batch Size', default=256)
    parser.add_argument('-d', '--gamma', type=float, help='Discount Factor', default=0.99)
    parser.add_argument('-c', '--critic_lr', default=3e-4, type=float, help='Critic Learning_rate')  
    parser.add_argument('-a', '--actor_lr', default=3e-4, type=float, help='Actor Learning_rate')  

    args = parser.parse_args() 
    al, cl, dc, bs = args.actor_lr, args.critic_lr, args.gamma, args.batch_size
    print("Start Training: actor/critic lr = %.4f / %.4f, batch size = %d, and gamma = %.2f" %(al,cl,dc,bs))
    print("#"*60)

    # batch size + discount + actor lr + critic lr
    log_path = os.path.join('./log_dir', "%s_%s_%s_%s"%(bs,dc,al,cl))
    logger.set_dir(log_path)
    tb_logger = LogWriter(log_path)
    main()
    
