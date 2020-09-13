import gym
import yaml  
import argparse
from rlschool import make_env
from train import Trainer

    
with open("config.yaml") as f:
    config = yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Test')
    parser.add_argument('--algo', type=str, default='ddpg')
    args = parser.parse_args()
    if args.algo == 'ddpg':
        from agents.ddpg import Agent
    elif args.algo == 'sac':
        from agents.sac import Agent
    elif args.algo == 'td3':
        from agents.td3 import Agent
    env = make_env("Quadrotor", task="hovering_control")
    game =  Trainer(config, env, Agent, args.algo)
    game.run_game()
    # print(yaml.SafeLoader.yaml_constructors)
