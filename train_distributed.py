# CUDA_VISIBLE_DEVICES=0,2,4,6 torchrun --nproc_per_node=4  train_distributed.py --num_classes 5 --model vgg
# 推荐阅读参考博文： https://blog.csdn.net/orangerfun/article/details/123887725
import os
import math
import shutil
import tempfile
import argparse
import random
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import classic_models
from dataload.dataload_five_flower import Five_Flowers_Load
from utils.train_engin import train_one_epoch, evaluate
from utils.distrubute_utils import clean_up, dist, init_distrubuted_mode 
from utils.lr_methods import warmup

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=100, help='the number of classes')
parser.add_argument('--epochs', type=int, default=100, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lrf', type=float, default=0.0001)
parser.add_argument('--seed', default=False, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization') 
parser.add_argument('--syncBN', type=bool, default=True, help='use syncBN during distrubute learning') 
parser.add_argument('--data_path', type=str,  default="/data/haowen_yu/code/dataset/flowers") 
parser.add_argument('--model', type=str, default="alexnet")
parser.add_argument('--weights', type=str, default='', help='initial weights path') 
parser.add_argument('--freeze_layers', type=bool, default=False) 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)') 
parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes') 
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
opt = parser.parse_args()

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed) # Python random module.	
        os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed) # Numpy module.
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # 实际上这个设置对精度影响不大，仅仅是小数点后几位的差别。所以如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低。
        print('random seed has been fixed')
    seed_torch() 


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distrubuted_mode(opt=args) 
    device = torch.device(args.device)   
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

    if opt.tensorboard:
        # 这是存放你要使用tensorboard显示的数据的绝对路径 
        log_path = os.path.join('/data/haowen_yu/code/classification/results/tensorboard' , args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path)) 
        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path) #当log文件存在时删除文件夹。记得在代码最开始import shutil 
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

    train_dataset = Five_Flowers_Load(os.path.join(args.data_path , 'train'), transform=data_transform["train"])
    val_dataset = Five_Flowers_Load(os.path.join(args.data_path , 'val'), transform=data_transform["val"]) 
 
    if args.num_classes != train_dataset.num_class:
        raise ValueError("dataset have {} classes, but input {}".format(train_dataset.num_class, args.num_classes))
 

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    
    if args.rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
        # save parameters path
    save_path = os.path.join(os.getcwd(), 'results/weights', args.model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    # 实例化模型
    model = classic_models.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device) 

    # 如果存在预训练权重则载入
    if os.path.exists(args.weights):
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if args.rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    optimizer = optim.Adam(pg, lr=args.lr)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch, lr_method=warmup)
        scheduler.step()

        # validate
        val_acc = evaluate(model=model, data_loader=val_loader, device=device)
        if args.rank == 0: 
            print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, mean_loss, train_acc, val_acc))   
        if opt.tensorboard:
            tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        # 判断当前验证集的准确率是否是最大的，如果是，则更新之前保存的权重
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, "AlexNet.pth")) 
 
    # 删除临时缓存文件
    if args.rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    clean_up()


if __name__ == '__main__':


    main(opt)