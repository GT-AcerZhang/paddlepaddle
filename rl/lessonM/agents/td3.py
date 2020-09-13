import numpy as np
import parl
from parl import layers
from paddle import fluid
from parl.algorithms import TD3


class Agent(parl.Agent):
    def __init__(self, obs_dim, act_dim, max_action, gamma, tau, actor_lr, critic_lr):
        model = Model(act_dim, max_action)
        self.alg = TD3(model, max_action, gamma, tau, actor_lr, critic_lr)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(self.alg)

        # 注意：最开始先同步self.model和self.target_model的参数.
        self.alg.sync_target(decay=0)
        self.learn_it = 0
        self.policy_freq = self.alg.policy_freq

    def build_program(self):
        self.pred_program = fluid.Program()
        self.actor_learn_program = fluid.Program()
        self.critic_learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.actor_learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.actor_cost = self.alg.actor_learn(obs)

        with fluid.program_guard(self.critic_learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.critic_cost = self.alg.critic_learn(obs, act, reward,next_obs, terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        act = np.squeeze(act)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        self.learn_it += 1
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.critic_learn_program,
            feed=feed,
            fetch_list=[self.critic_cost])

        actor_cost = None
        if self.learn_it % self.policy_freq == 0:
            actor_cost = self.fluid_executor.run(
                self.actor_learn_program,
                feed={'obs': obs},
                fetch_list=[self.actor_cost])[0]
            self.alg.sync_target()
        return critic_cost[0], actor_cost

    def save_model4visual(self, dir):
        feed_vars = ['obs','act','reward','next_obs','terminal']
        target_vars = [self.critic_cost]
        executor = self.fluid_executor
        program = self.critic_learn_program
        fluid.io.save_inference_model(dir, feed_vars, target_vars, executor, program, program_only=True)

    # def save(self, save_path, mode='actor'):
    #     if mode == 'actor':
    #         save_program=self.actor_learn_program
    #     elif mode == 'critic':
    #         save_program=self.critic_learn_program
    #     elif mode == 'predict':
    #         save_program=self.pred_program
    #     super(Agent, self).save(save_path, save_program)

    # def restore(self, save_path, mode='actor'):
    #     if mode == 'actor':
    #         load_program=self.actor_learn_program
    #     elif mode == 'critic':
    #         load_program=self.critic_learn_program
    #     elif mode == 'predict':
    #         load_program=self.pred_program
    #     super(Agent, self).restore(save_path, load_program)

class Model(parl.Model):
    def __init__(self, act_dim, max_action):
        self.actor_model = ActorModel(act_dim, max_action)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def Q1(self, obs, act):
        return self.critic_model.Q1(obs, act)

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

        self.fc4 = layers.fc(size=hid1_size, act='relu')
        self.fc5 = layers.fc(size=hid2_size, act='relu')
        self.fc6 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        hid2 = self.fc4(obs)
        concat2 = layers.concat([hid2, act], axis=1)
        Q2 = self.fc5(concat2)
        Q2 = self.fc6(Q2)
        Q2 = layers.squeeze(Q2, axes=[1])
        return Q1, Q2

    def Q1(self, obs, act):
        hid1 = self.fc1(obs)
        concat1 = layers.concat([hid1, act], axis=1)
        Q1 = self.fc2(concat1)
        Q1 = self.fc3(Q1)
        Q1 = layers.squeeze(Q1, axes=[1])

        return Q1