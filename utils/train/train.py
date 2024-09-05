import torch
from torch.optim import Adam
from models.multi_teacher_model import MultiTeacherModel
from models.lightweight_student_model import LightweightStudentModel
from utils.loss import MultiTeacherKnowledgeDistillationLoss
from utils.dataset import PlaceRecognitionDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def train(data_dir, epochs=10, batch_size=32, lr=0.001):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = PlaceRecognitionDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    teacher_models = MultiTeacherModel().cuda()
    student_model = LightweightStudentModel().cuda()
    
    criterion = MultiTeacherKnowledgeDistillationLoss()
    optimizer = Adam(student_model.parameters(), lr=lr)

    teacher_models.eval()   

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        
        for images in dataloader:
            images = images.cuda()

            with torch.no_grad():
                teacher_output1 = teacher_models.teacher1(images)
                teacher_output2 = teacher_models.teacher2(images)
                teacher_output3 = teacher_models.teacher3(images)

            teacher_outputs = [teacher_output1, teacher_output2, teacher_output3]
            student_output = student_model(images)
            labels = torch.randint(0, 2, (batch_size,)).cuda()  # 假设二分类标签
            
            loss = criterion(student_output, teacher_outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}')

if __name__ == "__main__":
    train(data_dir="path/to/data", epochs=10, batch_size=16)
