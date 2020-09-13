import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from utils import plot_data


# from utils import draw_process

if __name__ == "__main__":
    directory_path = './log_dir/results'
    file_type = 'npz'
    np_vars = dict()
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))
    for file in listdir(directory_path):
        name, post = file.split('.')
        if post == file_type:
            with np.load(directory_path+'/{}'.format(file),allow_pickle=True) as data:
                np_vars[name] = dict(zip((k for k in data.files), (data[k] for k in data.files)))
    titles = [('Train Reward',['train'],"#0000FF"), ('Test Reward',['mean','std'],"#1F618D"), ('Actor Loss',['aloss'],"#7FB3D5"), ('Critic Loss',['closs'],"#22DAF3")]
    for j, alg in enumerate(np_vars):
        agg_data = np_vars[alg]
        axes[0][j].set_title(alg)
        for i, row in enumerate(axes):
            title, keys, color = titles[i]
            plot_data(row[j], title, color, *(agg_data.get(key) for key in keys))
            if j == 0:
                row[0].set_ylabel(title)
    fig.tight_layout()
    plt.show()