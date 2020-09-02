from model import MNIST, MNIST_CNN
from preprocess import load_data
import paddle
import paddle.fluid as fluid
import os


# it has 50k training data, 10k validation data and 10k test data

BATCH_SIZE = 100
EPOCH_NUM = 5


def main():
    with fluid.dygraph.guard(place=fluid.CPUPlace()):
        
        model = MNIST_CNN('mnist_model')
        model.train()
        train_loader = load_data('train', BATCH_SIZE)
        
        # 定义学习率，并加载优化器参数到模型中
        total_steps = (int(60000//BATCH_SIZE) + 1) * EPOCH_NUM
        lr = fluid.dygraph.PolynomialDecay(0.01, total_steps, 0.001)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=model.parameters())
        
        for epoch_id in range(EPOCH_NUM):

            for batch_id, data in enumerate(train_loader()):

                image_data, label_data = data
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)

                # forward propagation
                predict, acc = model(image, label)                
                loss = fluid.layers.cross_entropy(predict, label)
                avg_loss = fluid.layers.mean(loss)
                avg_acc  = fluid.layers.mean(acc)
                if batch_id % 200 == 0:
                    print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),avg_acc.numpy()))
                
                # backward propagation
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()
        
        # save model and optim @para
        fluid.save_dygraph(model.state_dict(), './checkpoint/epoch{}'.format(epoch_id))
        fluid.save_dygraph(optimizer.state_dict(), './checkpoint/epoch{}'.format(epoch_id))

if __name__ == "__main__":
    main()