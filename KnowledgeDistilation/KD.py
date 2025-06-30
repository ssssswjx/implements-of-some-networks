import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

num_classes = 10
num_epochs = 6
batch_size = 32

class TeacherModel(nn.Module):
    def __init__(self, num_classes):
        super(TeacherModel, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.relu(self.dropout(self.fc3(x)))

        return x

class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc3 = nn.Linear(20, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc3(x)

        return x


transfrom = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST('C:\\Users\\wjxuan\\Documents\\PaperCode_implement\\mnist_data', train=True, download=True, transform=transfrom)
test_dataset = datasets.MNIST('C:\\Users\\wjxuan\\Documents\\PaperCode_implement\\mnist_data', train=False, download=True, transform=transfrom)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

teacher_model = TeacherModel(num_classes).to(device)
student_model = StudentModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher_model.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
optimizer_student = optim.Adam(student_model.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

# TeacherModel training
for epoch in range(num_epochs):
    teacher_model.train()
    for idx, data in enumerate(tqdm(train_loader, position=0, leave=True)):
        image, label = data
        image = image.to(device)
        label = label.to(device)

        pred = teacher_model(image)
        pred_image = torch.argmax(pred, dim=1)

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    teacher_model.eval()
    num_correct = 0
    total_num = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            image, label = data
            image = image.to(device)
            label = label.to(device)

            preds = teacher_model(image)
            pred_iamge = torch.argmax(preds, dim=1)

            num_correct += (pred_iamge == label).sum()
            total_num += len(pred_iamge)

        acc = num_correct / total_num

    print(f'\nEpoch :{epoch+1}\t TeacherModel   Accuracy:{acc:.4f}')


# StudentModel training
for epoch in range(3):
    student_model.train()
    for idx, data in enumerate(tqdm(train_loader, position=0, leave=True)):
        image, label = data
        image = image.to(device)
        label = label.to(device)

        pred = student_model(image)
        pred_image = torch.argmax(pred, dim=1)

        loss = criterion(pred, label)

        optimizer_student.zero_grad()
        loss.backward()
        optimizer_student.step()

    student_model.eval()
    num_correct = 0
    total_num = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            image, label = data
            image = image.to(device)
            label = label.to(device)

            preds = student_model(image)
            pred_iamge = torch.argmax(preds, dim=1)

            num_correct += (pred_iamge == label).sum()
            total_num += len(pred_iamge)

        acc = num_correct / total_num

    print(f'\nEpoch :{epoch+1}\t StudentModel Accuracy:{acc:.4f}')


#  konwledge distillation
T = 7
hard_loss = nn.CrossEntropyLoss()
soft_loss = nn.KLDivLoss(reduction='batchmean')
alpha = 0.3

optimizer = optim.Adam(student_model.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

teacher_model.eval()
for epoch in range(num_epochs):
    student_model.train()
    for image, label in tqdm(train_loader, position=0, leave=True):
        image = image.to(device)
        label = label.to(device)
        pred_student = student_model(image)
        student_loss = hard_loss(pred_student, label)

        with torch.no_grad():
            pred_teacher = teacher_model(image)

        distilled_loss = soft_loss(F.softmax(pred_student / T, dim=1), F.softmax(pred_teacher / T, dim=1)) * T * T

        loss = student_loss*alpha + distilled_loss*(1 - alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student_model.eval()
    num_correct = 0
    total_num = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            image, label = data
            image = image.to(device)
            label = label.to(device)

            preds = student_model(image)
            pred_iamge = torch.argmax(preds, dim=1)

            num_correct += (pred_iamge == label).sum()
            total_num += len(pred_iamge)

        acc = num_correct / total_num

    print(f'\nEpoch :{epoch+1}\t StudentModel Accuracy after distillation:{acc:.4f}')
