import os 
import argparse 
import math
import shutil
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from cnn_models.alexnet import AlexNet 
from utils.lr_methods import warmup
from dataload.dataload_mini_imagenet import MyDataSet
from dataload.dataload_five_flower import Five_Flowers_Load
from utils.train_engin import train_one_epoch, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lrf', type=float, default=0.00002)

# 数据集所在根目录
parser.add_argument('--data-path', type=str, default="/data/haowen_yu/code/dataset/flowers")

parser.add_argument('--weights', type=str, default='', help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

opt = parser.parse_args()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)

    # 这是存放你要使用tensorboard显示的数据的绝对路径
    log_path =  '/data/haowen_yu/code/results/tensorboard/alexnet'
    print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path)) 
    try:
        shutil.rmtree(log_path) #当log文件存在时删除文件夹。记得在代码最开始import shutil
        print("The folder has been emptied.")
    except:
        print("The folder does not exist.") #当log文件不存在时，直接打印“文件夹不存在”。
    # 实例化一个tensorboard
    tb_writer = SummaryWriter(log_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = args.data_path
    # json_path = '/data/haowen_yu/code/dataset/mini-imagenet/classes_name.json'

    # 实例化训练数据集
    # train_dataset = MyDataSet(root_dir=data_root, csv_name="new_train.csv", json_path=json_path, transform=data_transform["train"])
    # val_dataset = MyDataSet(root_dir=data_root, csv_name="new_val.csv", json_path=json_path, transform=data_transform["val"])

    train_dataset = Five_Flowers_Load(os.path.join(data_root, 'train'), transform=data_transform["train"])
    val_dataset = Five_Flowers_Load(os.path.join(data_root, 'val'), transform=data_transform["val"])

    if args.num_classes != train_dataset.num_class:
        raise ValueError("dataset have {} classes, but input {}".format(len(train_dataset.labels), args.num_classes))

    # 实例化验证数据集

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,  num_workers=nw, collate_fn=val_dataset.collate_fn)

    # create model
    model = AlexNet(num_classes=5).to(device) 

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    optimizer = optim.Adam(pg, lr=args.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch, lr_method=None)
        scheduler.step()

        # validate
        acc = evaluate(model=model, data_loader=val_loader, device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], acc, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        if os.path.exists("/data/haowen_yu/code/results/weights/alexnet") is False:
            os.makedirs("/data/haowen_yu/code/results/weights/alexnet")
        torch.save(model.state_dict(), "/data/haowen_yu/code/results/weights/alexnet/model-{}.pth".format(epoch))


if __name__ == '__main__':


    main(opt)