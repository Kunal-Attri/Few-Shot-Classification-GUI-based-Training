import os
import shutil
import zipfile

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as Datasets
from PIL import Image
from easyfsl.samplers.task_sampler import TaskSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

st.title("Train Few Shot classification models in Browser")
col1, col2 = st.columns(2)
test_box = None
test_image = None
image = None
train = None
train_loader = None
modelAvailable = False
data_path = None


def isModelAvailable():
    global modelAvailable
    modelAvailable = os.path.exists("Few_shot_model.pth.tar")


def removeFiles():
    if os.path.exists("train"):
        shutil.rmtree("train")
    if os.path.exists("Few_shot_model.pth.tar"):
        os.remove("Few_shot_model.pth.tar")


isModelAvailable()


def load_img(image_to_load):
    img = Image.open(image_to_load)
    return img


class Network(nn.Module):
    def __init__(self, pretrained_model):
        super(Network, self).__init__()
        self.pretrained_model = pretrained_model

    def forward(self, support_imgs, support_labels, query_imgs):
        z_support = self.pretrained_model.forward(support_imgs)
        z_query = self.pretrained_model.forward(query_imgs)
        num_classes = len(support_labels.unique())
        z_proto = torch.cat([
            z_support[torch.nonzero(support_labels == i)].mean(0)
            for i in range(num_classes)
        ])
        dist = torch.cdist(z_query, z_proto)

        return -dist


path = st.file_uploader("Input Dataset", type="zip")
split_ratio = col1.slider("Train-Test split ratio") / 100
image_size = st.number_input("Image size for Data augmentation", step=1, min_value=100, max_value=512)

transform = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor()
])

n_way = col2.slider("Unique Classes in the dataset", max_value=40, min_value=2)
n_shot = col2.number_input("Count of Images in each Class of Support Set", step=1)
n_query = col2.number_input("Count of Images in each Class of Query Set", step=1)
train_tasks = col1.number_input("Episodes in the Train Set", step=1)
test_tasks = col1.number_input("Episodes in the Test Set", step=1)
learning_rate = 3e-4

if path is not None:
    open('tempzip.zip', 'wb').write(path.getvalue())
    with zipfile.ZipFile("tempzip.zip", "r") as zip_ref:
        zip_ref.extractall("")
    os.remove("tempzip.zip")
    data_path = path.name[:-4]
else:
    removeFiles()

if data_path is not None:
    data = Datasets.ImageFolder(root=data_path, transform=transform)
    split_list = [int(split_ratio * len(data)), len(data) - int(split_ratio * len(data))]
    train_set, test_set = torch.utils.data.random_split(dataset=data, lengths=split_list)

    train_set.get_labels = lambda: [i[1] for i in train_set]
    train_sampler = TaskSampler(train_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=train_tasks)
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler,
                              collate_fn=train_sampler.episodic_collate_fn, pin_memory=True)

    test_set.get_labels = lambda: [i[1] for i in test_set]
    test_sampler = TaskSampler(dataset=test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=test_tasks)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler,
                             collate_fn=test_sampler.episodic_collate_fn, pin_memory=True)

backbone = resnet18(pretrained=True)
backbone.fc = nn.Flatten()
model = Network(pretrained_model=backbone)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_model():
    loss_list = []
    for i, (support_images, support_labels, query_images, query_labels, _) in enumerate(train_loader):
        out = model(support_images, support_labels, query_images)
        loss = criterion(out, query_labels)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % (len(train_loader) * 0.25) == 0 and i + 1 >= (len(train_loader) * 0.25):
            st.write(100 * (i + 1) / len(train_loader), "% Training completed", "Training loss:", loss,
                     " Model weights saved")
            torch.save({
                'optim_state_dict': optimizer.state_dict(),
                "loss": loss,
                "model_state_dict": model.state_dict(),
                "Episode_num": i
            }, "Few_shot_model.pth.tar")
    shutil.copy2("Few_shot_model.pth.tar", "model.pth.tar")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(loss_list)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Training Loss")

    st.write(fig)
    isModelAvailable()


if train_loader is not None:
    train = st.button("Train model")
if train:
    train_model()

if modelAvailable:
    checkpoint = torch.load("Few_shot_model.pth.tar", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    episode_num = checkpoint['Episode_num']
    loss = checkpoint['loss']


def evaluate():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (support_images, support_labels, query_images, query_labels, _) in enumerate(test_loader):
            out = model(support_images, support_labels, query_images)
            correct += (torch.max(out, 1)[1] == query_labels).sum().item()
            total += len(query_labels)
        st.write(f"Model tested on {len(test_loader)} tasks with Accuracy of {correct * 100 / total} %")


if modelAvailable:
    test_box = st.button("Evaluate Model on Test set")

if test_box:
    evaluate()

if modelAvailable:
    image = st.file_uploader("*Model's Output for a single image*")

if image is not None:
    image1 = load_img(image)
    # print(image1.shape)
    for i, (support_images, support_labels, _, query_labels, _) in enumerate(train_loader):
        image1 = transform(image1.convert('RGB'))
        image1 = image1.repeat(n_way * n_query, 1, 1, 1)
        out = model(support_images, support_labels, image1)
        st.write("Image belongs to class: " + str(max((torch.max(out, 1)[1])).item()))
        break

if modelAvailable:
    st.download_button("Download Model",
                       data=open("model.pth.tar", 'rb'))
