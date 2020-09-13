import numpy as np
import parl
from parl import layers
import paddle.fluid as fluid
from parl.algorithms import DDPG


class Agent(parl.Agent):
    def __init__(self, obs_dim, act_dim, max_action, gamma, tau, actor_lr, critic_lr):
        model = Model(act_dim, max_action)
        self.alg = DDPG(model, gamma, tau, actor_lr, critic_lr)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(self.alg)

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.actor_cost, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        act = np.squeeze(act)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        [critic_cost, actor_cost] = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost, self.actor_cost])
        self.alg.sync_target()
        return critic_cost[0], actor_cost[0]

    def save_model4visual(self, dir, name):
        feed_vars = ['obs','act','reward','next_obs','terminal']
        target_vars = [self.actor_cost, self.critic_cost]
        executor = self.fluid_executor
        program = self.learn_program
        fluid.io.save_inference_model(dir, feed_vars, target_vars, executor, program, program_only=True)

class Model(parl.Model):
    def __init__(self, act_dim, max_action):
        self.actor_model = ActorModel(act_dim, max_action)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim, max_action):
        hid1_size = 400
        hid2_size = 300
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')
        self.max_action = max_action

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        means = self.fc3(hid2)
        return means * self.max_action


class CriticModel(parl.Model):
    def __init__(self):
        hid1_size = 400
        hid2_size = 300

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        hid1 = self.fc1(obs)
        concat = layers.concat([hid1, act], axis=1)
        hid2 = self.fc2(concat)
        Q = self.fc3(hid2)
        Q = layers.squeeze(Q, axes=[1])
        return Q