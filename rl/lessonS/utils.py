import numpy as np
from parl.utils import action_mapping

WARMUP_SIZE = 1e4
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise


def run_episode(env, agent, rpm, batch_size=64):
    obs = env.reset()
    total_reward, steps, a_loss, c_loss = 0, 0, 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        action = np.clip(np.random.normal(action, EXPL_NOISE), -1.0, 1.0)
        action = action_mapping(action, env.action_space.low[0], env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)

        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                batch_size)
            a_loss, c_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps, a_loss, c_loss

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    obs = env.reset()
    total_reward = 0
    while True:
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        action = np.clip(action, -1.0, 1.0)  ## special
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward

