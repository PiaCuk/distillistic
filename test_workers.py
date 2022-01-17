import os
import time
import torch
from torchvision import datasets, transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pin_memory = True
persistent_workers = True
print(f"Hyperparams: pin_memory is {pin_memory}, persistent_workers is {persistent_workers}.")

data_path = "/home/pcuk/data/imagecat"
data_dir = os.path.join(data_path, 'train')
# Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        normalize,
    ])

train_data = datasets.ImageFolder(data_dir, trans)

for num_workers in [12]: # range(10, 17, 1):
    for b in [32, 64, 128]:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=b,
                                                   num_workers=num_workers, pin_memory=pin_memory,
                                                   persistent_workers=True)
        start = time.time()
        for epoch in range(1, 5):
            for i, data in enumerate(train_loader):
                pass
        end = time.time()
        print("Finish with:{} second, num_workers={}, batch_size={}".format(
            end - start, num_workers, b))