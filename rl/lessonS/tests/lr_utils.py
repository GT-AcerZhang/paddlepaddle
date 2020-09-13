import math
import paddle.fluid as fluid
import numpy as np

def exponential_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    exponent = global_step / decay_steps
    if staircase:
        exponent = math.floor(exponent)
    return learning_rate * decay_rate**exponent


def natural_exp_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    exponent = float(global_step) / float(decay_steps)
    if staircase:
        exponent = math.floor(exponent)
    return learning_rate * math.exp(-1 * decay_rate * exponent)


def inverse_time_decay(learning_rate,
                       global_step,
                       decay_steps,
                       decay_rate,
                       staircase=False):
    temp = float(global_step) / float(decay_steps)
    if staircase:
        temp = math.floor(temp)
    return learning_rate / (1 + decay_rate * temp)


def polynomial_decay(learning_rate,
                     global_step,
                     decay_steps,
                     end_learning_rate=0.0001,
                     power=1.0,
                     cycle=False):
    if cycle:
        div = math.ceil(global_step / float(decay_steps))
        if div == 0:
            div = 1
        decay_steps = decay_steps * div
    else:
        global_step = min(global_step, decay_steps)
    return (learning_rate - end_learning_rate) * \
           ((1 - float(global_step) / float(decay_steps)) ** power) + end_learning_rate


def piecewise_decay(global_step, boundaries, values):
    assert len(boundaries) + 1 == len(values)
    for i in range(len(boundaries)):
        if global_step < boundaries[i]:
            return values[i]
    return values[len(values) - 1]


def cosine_decay(global_step, learning_rate, step_each_epoch, epochs):
    cur_epoch = math.floor(global_step / step_each_epoch)
    decayed_lr = learning_rate * 0.5 * (
        math.cos(cur_epoch * math.pi / epochs) + 1)
    return decayed_lr


def noam_decay(global_step, d_model, warmup_steps, learning_rate=1.0):
    a = math.pow(global_step, -0.5)
    b = math.pow(warmup_steps, -1.5) * global_step
    decayed_lr = learning_rate * math.pow(d_model, -0.5) * min(a, b)

    return decayed_lr


def linear_lr_warmup(global_step, warmup_steps, start_lr, end_lr):
    linear_step = end_lr - start_lr
    decayed_lr = start_lr + linear_step * (global_step / warmup_steps)
    return decayed_lr


def multi_step_decay(global_step, learning_rate, milestones, decay_rate=0.1):
    for i in range(len(milestones)):
        if global_step < milestones[i]:
            return learning_rate * math.pow(decay_rate, i)

    return learning_rate * math.pow(decay_rate, len(milestones))


def step_decay(global_step, learning_rate, step_size, decay_rate=0.1):
    return learning_rate * math.pow(decay_rate, global_step // step_size)


def lambda_decay(global_step, learning_rate, lr_lambda):
    return learning_rate * lr_lambda(global_step)

"""
decay_fns = [(exponential_decay, lr_.exponential_decay, {"learning_rate": 0.1,"decay_steps": 20,"decay_rate": 0.5,"staircase": True}),
             (natural_exp_decay, lr_.natural_exp_decay, {"learning_rate": 0.1,"decay_steps": 20,"decay_rate": 0.5,"staircase": True}),
             (inverse_time_decay, lr_.inverse_time_decay, {"learning_rate": 0.1,"decay_steps": 20,"decay_rate": 0.5,"staircase": True}),
             (polynomial_decay, lr_.polynomial_decay,   {"learning_rate": 0.1,"decay_steps": 20,"cycle": True}),
             (piecewise_decay, lr_.piecewise_decay,     {"boundaries": [50,100,150], "values":[0.1,0.075,0.05,0.01]}),
             (cosine_decay, lr_.cosine_decay,           {"learning_rate": 0.1,"step_each_epoch": 20,"epochs": 10}),
             (noam_decay, lr_.noam_decay,               {"warmup_steps": 20, "d_model": 2})]
df = gen_lr(decay_fns, nsteps=200)
"""

def gen_lr(decay_fns, nsteps=20):
    df = dict()
    for py_decay_fn, fluid_decay_fn, kwargs in decay_fns:
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            decayed_lr = fluid_decay_fn(**kwargs)
        name = py_decay_fn.__name__.split('_')[0]
        df[name] = np.zeros(nsteps)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_prog)
        for step in range(nsteps):
            lr_val = exe.run(main_prog, feed={}, fetch_list=decayed_lr)[0]
            if name == 'noam':
                py_decayed_lr = lr_val
            else:
                py_decayed_lr = py_decay_fn(global_step=float(step), **kwargs)
            # assertAlmostEqual(py_decay_lr, lr_val, "Python result{} and Fluid result is {}".format(py_decay_lr, lr_val))
            df[name][step] = lr_val
    return df