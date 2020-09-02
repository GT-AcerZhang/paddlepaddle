import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Conv2D, Pool2D, Linear


class MNIST_(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc = Linear(input_dim=784, output_dim=1, act=None)
    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, (-1, 784))
        outputs = self.fc(inputs)
        return outputs

class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        # 定义两层全连接隐含层，输出维度是10，激活函数为sigmoid
        self.fc1 = Linear(input_dim=784, output_dim=10, act='sigmoid') # 隐含层节点为10，可根据任务调整
        self.fc2 = Linear(input_dim=10, output_dim=10, act='sigmoid')
        # 定义一层全连接输出层，输出维度是1，不使用激活函数
        self.fc3 = Linear(input_dim=10, output_dim=1, act=None)
    
    # 定义网络的前向计算
    def forward(self, inputs, label=None):
        inputs = fluid.layers.reshape(inputs, [inputs.shape[0], 784])
        outputs1 = self.fc1(inputs)
        outputs2 = self.fc2(outputs1)
        outputs_final = self.fc3(outputs2)
        return outputs_final

class MNIST_CNN(fluid.dygraph.Layer):
     def __init__(self, name_scope):
         super(MNIST_CNN, self).__init__(name_scope)
         name_scope = self.full_name()
         # 定义一个卷积层，使用relu激活函数
         self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个卷积层，使用relu激活函数
         self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个全连接层，输出节点数为10 
         self.fc = Linear(input_dim=980, output_dim=10, act='softmax')
    # 定义网络的前向计算过程
     def forward(self, inputs, label):
         x = self.conv1(inputs)
         x = self.pool1(x)
         x = self.conv2(x)
         x = self.pool2(x)
         x = fluid.layers.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         if label is not None:
             acc = fluid.layers.accuracy(input=x, label=label)
             return x, acc
         else:
             return x