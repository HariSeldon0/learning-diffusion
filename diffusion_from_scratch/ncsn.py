import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------
# 1. 分布式训练初始化
# ---------------------------
def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device('cuda', local_rank)

# ---------------------------
# 2. 数据准备 (支持 DistributedSampler)
# ---------------------------
def get_dataloaders(batch_size, dataset='CIFAR10'):
    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Only CIFAR10 is implemented")

    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader, train_sampler

# ---------------------------
# 3. 模型定义 (保持原样)
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.block(x) + self.shortcut(x))

class ScoreCNN(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(64, 128),
            nn.AvgPool2d(2),
            ResidualBlock(128, 128),
            nn.AvgPool2d(2),
            ResidualBlock(128, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(128, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(128, 64),
            nn.Conv2d(64, channels, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# 4. Checkpoint 保存/恢复
# ---------------------------
def save_checkpoint(model, optimizer, epoch, path='checkpoint.pt'):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, path)

def load_checkpoint(model, optimizer, path='checkpoint.pt'):
    if os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        return state['epoch'] + 1
    return 0

# ---------------------------
# 5. 训练循环
# ---------------------------
def train():
    local_rank, device = setup_distributed()
    batch_size = 512
    train_loader, test_loader, train_sampler = get_dataloaders(batch_size)

    model = ScoreCNN().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_epoch = load_checkpoint(model, optimizer, 'v2-checkpoint.pt')

    epoch_nums = 500
    noise_scale = 0.01

    for epoch in range(start_epoch, epoch_nums):
        train_sampler.set_epoch(epoch)
        for idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_noise = x + noise_scale * torch.randn_like(x)
            score = model(x_noise)
            target = (x - x_noise) / (noise_scale ** 2)
            loss = ((score - target) ** 2).sum(dim=(1, 2, 3)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if local_rank == 0:
            print(f"Epoch {epoch} - Loss {loss.detach().item()}")
        if local_rank == 0 and epoch % 50 == 0:  # 仅在 rank0 保存 checkpoint 和打印
            save_checkpoint(model, optimizer, epoch, 'v2-checkpoint.pt')

if __name__ == '__main__':
    train()
