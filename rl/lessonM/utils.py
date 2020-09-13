import matplotlib.pyplot as plt
import numpy as np

class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self, entries):
        self.__dict__.update(**entries)

def agent2color(agent_name):
        """Creates a dictionary that maps an agent to a hex color (for plotting purposes)
        """
        colormap = {
            "DQN": "#0000FF",
            "DQN with Fixed Q Targets": "#1F618D",
            "DDQN": "#2980B9",
            "DDQN with Prioritised Replay": "#7FB3D5",
            "Dueling DDQN": "#22DAF3",
            "PPO": "#5B2C6F",
            "DDPG": "#800000",
            "DQN-HER": "#008000",
            "DDPG-HER": "#008000",
            "TD3": "#E74C3C",
            "h-DQN": "#D35400",
            "SNN-HRL": "#800000",
            "A3C": "#E74C3C",
            "A2C": "#F1948A",
            "SAC": "#1C2833",
            "DIAYN": "#F322CD",
            "HRL": "#0E0F0F"
        }
        return colormap[agent_name]

def visual4all(stats, names, show_mean_n_std=False, show_each_run=False, color=None, ax=None, title=None, y_limits=None):
    """Visualises the results for one agent
    """
    if not ax: ax = plt.gca()
    return 1
    # if show_mean_and_std_range:



class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0
        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.cnt = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)
        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.cnt = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = (
                (self.means * self.cnt) + (new_data_mean * n)) / (self.cnt + n)
            self.vars = (((self.cnt * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) /
                         (self.cnt + n) - np.square(new_means))
            self.vars = np.maximum(
                0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.cnt += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means


def draw_process(directory):
    """directory where I save npz file with train and test record
    """
    with np.load(directory+"/record.npz") as data:
        train_mean = data['train']
        test_mean, test_std = data['mean'], data['std']
        plot_reward(train_mean)
        plot_reward(test_mean, test_std)


def plot_reward(mean, std=None, color='red'):
    title = 'Training' if std is None else 'Testing'
    n = len(mean); x = np.arange(n)
    plt.title(title, fontsize=24)
    plt.xlabel("episode", fontsize=20)
    plt.ylabel("reward", fontsize=20)
    plt.plot(x, mean,color=color) 
    if std is not None:
        plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
    plt.show()

def plot_data(ax, name, color, data, std=None):
    n = len(data); x = np.arange(n)
    ax.plot(x, data, color=color)
    if std is not None:
        plt.fill_between(x, data-std, data+std, color=color, alpha=0.2)
