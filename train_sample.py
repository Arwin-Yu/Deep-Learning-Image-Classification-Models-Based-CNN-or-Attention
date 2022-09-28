import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
# 将创建AlexNet模型的python脚本导入进来
from classic_models.alexnet import AlexNet 

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载一种图片用于推理
    img_path = "/data/haowen_yu/code/dataset/flowers/val/daisy/3640845041_80a92c4205_n.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # 扩张一个batch维度，因为训练模型时使用的小批量随机梯度下降有batch维度，所以推理时也需要有
    img = torch.unsqueeze(img, dim=0)

    # 加载模型预测值与真实类别的对应关系，json文件详见我的github代码
    json_path = '/data/haowen_yu/code/dataset/flowers/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 实例化模型
    model = AlexNet(num_classes=5).to(device)

    # 加载模型的权重
    weights_path = "/data/haowen_yu/code/results/weights/alexnet/AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        # 取分类可能性最大的类别作为模型的识别结果
        predict_cla = torch.argmax(predict).numpy()

    # 以图片的方式输出识别结果
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
