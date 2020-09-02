import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from model import MNIST, MNIST_CNN
import paddle.fluid as fluid

# 失败了，等我学完了CNN的原理再回过头来解决dimension不匹配的问题


img_path = './example_0.png'
params_file_path = 'mnist'
IMG_ROWS, IMG_COLS = 28, 28

def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, IMG_ROWS, IMG_COLS).astype(np.float32)
    # 图像归一化，保持和数据集的数据范围一致
    im = 1 - im / 127.5
    return im

def main():
    use_gpu = False
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = MNIST_CNN()
        # 加载模型参数
        model_dict, _ = fluid.load_dygraph("mnist")
        model.load_dict(model_dict)
        # 灌入数据
        model.eval()
        tensor_img = load_image(img_path)
        result = model(fluid.dygraph.to_variable(tensor_img))
        #  取概率最大的标签作为预测输出
        output = np.argsort(result.numpy())
        print(output)
        print("本次预测的数字是", output[0][-1])


if __name__ == "__main__":
    main()