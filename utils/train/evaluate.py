import torch
from torch.utils.data import DataLoader
from models.lightweight_student_model import LightweightStudentModel
from utils.dataset import PlaceRecognitionDataset
from torchvision import transforms

def evaluate(data_dir, batch_size=16, checkpoint=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = PlaceRecognitionDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LightweightStudentModel().cuda()

    if checkpoint is not None:
        print(f"Loading model from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint))

    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images in dataloader:
            images = images.cuda()
            outputs = model(images)

            _, top5_indices = outputs.topk(5, dim=1)
            _, top1_index = outputs.topk(1, dim=1)

            # correct_top1 += torch.sum(top1_index == ground_truth_labels).item()
            # correct_top5 += torch.sum(top5_indices == ground_truth_labels).item()
            total += images.size(0)

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total

    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")

if __name__ == "__main__":
    evaluate(data_dir="path/to/test/data", batch_size=16, checkpoint="path/to/model/checkpoint")
