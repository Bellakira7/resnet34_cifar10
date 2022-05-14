import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# 超参数定义
EPOCH =50
BATCH_SIZE = 64
LR = 0.01
# 数据集加载
# 对训练集及测试集数据的不同处理组合
transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                           ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
# 将数据加载进来，本地已经下载好， root=os.getcwd()为自动获取与源码文件同级目录下的数据集路径
train_data = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_data = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

# 数据分批
from torch.utils.data import DataLoader

# 使用DataLoader进行数据分批，dataset代表传入的数据集，batch_size表示每个batch有多少个样本
# shuffle表示在每个epoch开始的时候，对数据进行重新排序
# 数据分批之前：torch.Size([3, 32, 32])：Tensor[[32*32][32*32][32*32]],每一个元素都是归一化之后的RGB的值；数据分批之后：torch.Size([64, 3, 32, 32])
# 数据分批之前：train_data([50000[3*[32*32]]])
# 数据分批之后：train_loader([50000/64*[64*[3*[32*32]]]])
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 模型加载，有多种内置模型可供选择
model = torchvision.models.resnet34()
model.conv1=torch.nn.Conv2d(3,64,3,1,1)
model.maxpool=torch.nn.MaxPool2d(3,1,1)
model.avgpool=torch.nn.AvgPool2d(4)
model.fc=torch.nn.Linear(512,10)
# 定义损失函数，分类问题使用交叉信息熵，回归问题使用MSE
criterion = nn.CrossEntropyLoss()
# torch.optim来做算法优化,该函数甚至可以指定每一层的学习率，这里选用Adam来做优化器，还可以选其他的优化器
optimizer = optim.SGD(model.parameters(), lr=LR,weight_decay=0.001,momentum=0.9)
stepLR=torch.optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.1)

# 设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 模型和输入数据都需要to device
'''model_dict=model.state_dict()
Resnet18=torchvision.models.resnet18(pretrained=True)
Resnet18.fc=torch.nn.Linear(512,10)
pretrained_dict=Resnet18.state_dict()
model_list=list(model_dict.keys())
pretrained_list=list(pretrained_dict.keys())
len1=len(model_list)
len2=len(pretrained_list)
minlen=min(len1,len2)
for n in range(minlen):
    if model_dict[model_list[n]].shape!=pretrained_dict[pretrained_list[n]].shape:
        continue
    model_dict[model_list[n]]=pretrained_dict[pretrained_list[n]]'''
model.to(device)
model=torch.nn.DataParallel(model)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = True
# 模型训练
writer=SummaryWriter('./log')
temp=100
running_loss=0.0
for epoch in range(EPOCH):
    model.train()
    correct, total = 0, 0
    for i, data in enumerate(train_loader):
        # 取出数据及标签
        inputs, labels = data
        # 数据及标签均送入GPU或CPU
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        # 计算损失函数
        loss = criterion(outputs, labels)
        # 清空上一轮的梯度
        optimizer.zero_grad()


        # 反向传播
        loss.backward()
        running_loss+=loss.item()
        # 参数更新
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        if i % 49 == 0 and i!=0:
            print("epoch {} - iteration {}: average loss {:.3f}".format(epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    print('训练集准确率：{:.4f}%'.format(100.0 * correct / total))
    writer.add_scalar('train_acc',correct,epoch)
    stepLR.step()
    model.eval()
    correct, total = 0, 0
    count=0
    running_loss=0.0
    for j, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        count+=1
    print('测试集loss:{:.3f}.'.format(running_loss/count))
    print('测试集准确率：{:.4f}%'.format(100.0 * correct / total))
    writer.add_scalar('test_acc',correct,epoch)
    if (running_loss/count)<temp:
        torch.save(model, 'cifar10_resnet18.pt')
        temp=running_loss
    print('cifar10_resnet18.pt saved')
    running_loss=0.0
writer.close()
