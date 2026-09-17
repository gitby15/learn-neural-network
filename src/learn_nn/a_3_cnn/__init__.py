import torch
import tqdm as tqdm
from torch import nn

from learn_nn.a_3_cnn.dataset import get_test_dataset, get_train_dataset


class CNN(nn.Module):
    def __init__(
        self,
        num_kernels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_size: int,
        pool_stride: int,
        output_dim: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, num_kernels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(output_dim)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.flatten(x)
        return self.fc1(x)



def train() -> CNN:
    batch_size = 64
    num_epochs = 10

    model = CNN(
        num_kernels=8,
        kernel_size=3,
        stride=1,
        padding=1,
        pool_size=2,
        pool_stride=2,
        output_dim=10,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_progress = tqdm.tqdm(range(num_epochs), desc="Training")
    for epoch in epoch_progress:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        train_dataset = get_train_dataset(batch_size)
        batch_progress = tqdm.tqdm(train_dataset, desc=f"Epoch {epoch + 1}", leave=False)
        for image_batch, label_batch in batch_progress:
            images = torch.tensor(image_batch).reshape(-1, 1, 28, 28)
            labels = torch.tensor(label_batch)
            

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_samples = labels.size(0)
            total_loss += loss.item() * batch_samples
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_samples
            batch_progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            accuracy=f"{total_correct / total_samples:.2%}",
        )

    return model

def test(model: CNN) -> None:
    test_dataset = get_test_dataset(64)
    for image, label in test_dataset:
        images = torch.tensor(image).reshape(-1, 1, 28, 28)
        labels = torch.tensor(label)
        logits = model(images)
        predict_label = logits.argmax(dim=1)
        print(f"predict_label: {predict_label}")
        print(f"labels       : {labels}")
        print("=" * 50)


if __name__ == '__main__':
    from learn_nn.utils.device import get_cached_device
    torch.set_default_device(get_cached_device())
    model = train()
    test(model)

# 原来还做了一个web server，在网页上手写数字，让训练后的模型来识别，虽然成功率很低，但是很有意思
# 当时debug的结论，是因为自己用canvas实现的数字，边缘的处理跟训练数据的边缘差异很大（训练数据的边缘长，更平滑，canvas画的边缘短）
# 后来魔改训练数据，强行把训练数据的边缘做短（类似做一次sigmod，小于125就设置以为0）
# 时间久远，重构项目后这个web server跑不起来了，因为这部分知识已经学会了，所以就不花时间修好它了
# 专门写这个注释来纪念当时在广州跟老婆一起玩手写数字识别的时光~