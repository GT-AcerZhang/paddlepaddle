import parl
from parl import layers  # 封装了 paddle.fluid.layers 的API


class Model_v0(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q

class Model_v1(parl.Model):
    def __init__(self, act_dim):
        hid_size = 100
        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        h = self.fc1(obs)
        Q = self.fc2(h)
        return Q

class Model_v2(parl.Model):
    def __init__(self, act_dim):
        self.fc1 = layers.fc(size=24, act='relu')
        self.fc2 = layers.fc(size=48, act='relu')
        self.fc3 = layers.fc(size=24, act='relu')
        self.fc4 = layers.fc(size=act_dim, act='relu')
    
    def value(self, obs):
        Q = self.fc4(self.fc3(self.fc2(self.fc1(obs))))
        return Q