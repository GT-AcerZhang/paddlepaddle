import paddle
import paddle.fluid as fluid
import numpy as np
import os
from PIL import Image
from model import MNIST
# import gzip
# import json


trainset = paddle.dataset.mnist.train()
train_reader = paddle.batch(trainset, batch_size=8)
EPOCH_NUM = 5

# for batch_id, data in enumerate(train_reader()):
#     img_data = np.array([x[0] for x in data]).astype('float32')
#     label_data = np.array([x[1] for x in data])
#     print("image shape = {0} and target shape = {1}".format(img_data, label_data))



if __name__ == "__main__":
    use_gpu = False
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = MNIST()                  # network
        model.train()                    # train mode
        train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)                 # reader -> batch
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=1e-3, parameter_list=model.parameters())

        for epoch_id in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                image_data = np.array([x[0] for x in data], dtype='float32')
                label_data = np.array([x[1] for x in data], dtype='float32')
                # in conventional setting, we use layers.data to construct graph first
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)
                # forward propagation
                predict = model(image)
                loss = fluid.layers.square_error_cost(predict, label)
                avg_loss = fluid.layers.mean(loss)
                # check loss for every 1000 batchs
                if batch_id !=0 and batch_id  % 1000 == 0:
                    print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy())) 
                # backpropagation
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()
        fluid.save_dygraph(model.state_dict(), 'mnist')