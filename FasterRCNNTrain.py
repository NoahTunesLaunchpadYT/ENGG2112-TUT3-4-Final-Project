import os
import time
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torch import nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from ArtificialImageDataset import ArtificialImageDataset
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class FasterRCNNTraining:
    def __init__(self, modelPath):
        print("Initializing model")
        num_classes = 2
        backbone = resnet_fpn_backbone(backbone_name='resnet18', weights=False)
        self.model = FasterRCNN(backbone, num_classes=num_classes)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Using device {self.device}')

        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=0.001)

        print("Model Initialized")
        print("Loading data")
        
        dataset = ArtificialImageDataset(r"C:\ENGG2110 project\ENGG2112-TUT3-4-Final-Project\TrainingImages")
        # test_dataset_visualization(dataset)

        print("Dataset size:", len(dataset))
        self.data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2)

        print("Initialized")

        print("Loading model")
        self.modelPath = modelPath
        try:
            self.loadModel(modelPath)
            print("Model loaded")
        except FileNotFoundError:
            print("Model not found, generating new model")
            self.saveModel("model.pth")
            try:
                os.remove("EpochLosses.txt")
                os.remove("BatchLosses.txt")
                os.remove("BatchInformation.txt")
            except FileNotFoundError:
                pass

        
    def trainOneEpoch(self, epoch):
        self.model.train()
        epochLoss = 0
        print(f"Starting epoch {epoch + 1}")
        epochStartTime = time.time()

        for i, (images, targets) in enumerate(self.data_loader):
            print(f"Starting batch {i + 1}")

            batchStartTime = time.time()

            cached = torch.cuda.memory_reserved()
            print(f"Reserved memory: {cached / 1024**3:.2f} GiB")

            print("Loading images and targets")
            # Ensure images are RGB and remove any unnecessary batch dimension if batch size is 1
            images = [image.squeeze(0).to(self.device, non_blocking=True, copy=False)[:3, :, :] if image.shape[0] == 1 else image.to(self.device, non_blocking=True, copy=False)[:3, :, :] for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Ensure all targets have at least one bounding box
            for target in targets:
                if 'boxes' not in target or target['boxes'].nelement() == 0:
                    target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros(1, dtype=torch.int64, device=self.device)  # Assuming '0' is a background or dummy class

            print("Running model")

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            if not isinstance(losses, torch.Tensor):
                raise TypeError(f"Expected 'losses' to be a tensor, but got {type(losses)}")

            if losses.numel() == 1:
                loss = losses.item()
            else:
                loss = losses.sum().item()

            epochLoss += loss

            print("Running backpropagation")
            losses.sum().backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

              # Check if GPU memory usage exceeds a threshold before clearing cache
            if torch.cuda.memory_reserved() / 1024**3 > 1:  # Threshold of 1 GiB, adjust as necessary
                torch.cuda.empty_cache()
                print("Cleared GPU cache")

            print(f"Batch took a time of: {round(time.time() - batchStartTime, 2)} seconds")
            print(f"Loss for batch: {loss}")

            with open("BatchInformation.txt", "a") as file:
                file.write(f"Epoch: {epoch + 1} Batch: {i + 1} Loss: {loss} Time: {round(time.time() - batchStartTime, 2)} Memory: {cached / 1024**3:.2f} GiB\n")

        with open("EpochInformation.txt", "a") as file:
            file.write(f"Epoch: {epoch + 1} Loss: {epochLoss} Total Time: {round(time.time() - epochStartTime, 2)}\n")

        print(f"Epoch finished with a total time of: {round(time.time() - epochStartTime, 2)} seconds")
        print(f"Total Loss: {epochLoss}\n")

        self.saveModel("model.pth")


    def trainEpochs(self, epochs):
        try:
            for epoch in range(epochs):
                self.trainOneEpoch(epoch)
        except KeyboardInterrupt:
            print("Training interrupted")
            self.saveModel("InterruptSave.pth")
            exit()

    def saveModel(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
    
    def loadModel(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    images = torch.stack(images, dim=0)

    return images, targets

def show_image_with_boxes(image, boxes):
    """
    Display an image with bounding boxes using matplotlib for better control over display.
    """
    # Convert tensor to image array if it's a tensor.
    if isinstance(image, torch.Tensor):
        # Convert from [C, H, W] to [H, W, C]
        image = image.permute(1, 2, 0).numpy()
        # Normalize the image for display purposes.
        image = (image - image.min()) / (image.max() - image.min())

    # Create a plot to display the image.
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Add rectangles for bounding boxes with labels.
    for box in boxes:
        rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2,
                         edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')  # Turn off axes for better visibility
    plt.show()

def test_dataset_visualization(dataset):
    for i in range(10):  # Test the first 10 images
        image, target = dataset[i]
        show_image_with_boxes(image, target['boxes'])

def main():
    torch.cuda.empty_cache()
    training = FasterRCNNTraining("model.pth")
    training.trainEpochs(600)

if __name__ == "__main__":
    main()
