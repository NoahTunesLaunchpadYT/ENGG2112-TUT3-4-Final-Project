import json
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet18_Weights
from PIL import Image, ImageDraw
from torch import nn
import numpy as np

class FasterRCNNPredict:
    def __init__(self, model_path: str):
        print("Initializing model")
        num_classes = 2
        backbone = resnet_fpn_backbone('resnet18', weights=ResNet18_Weights.DEFAULT)
        self.model = FasterRCNN(backbone, num_classes=num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model. Error: {e}")

    def load_image(self, file_path: str) -> torch.Tensor:
        image = Image.open(file_path).convert("RGB")  # Ensure 3 channels for RGB
        image = np.array(image, dtype=np.float32)
        image /= 4095  # Normalize 12-bit image assuming max value is 4095 for 12-bit depth
        image_tensor = torch.tensor(image).permute(2, 0, 1)  # Reorder dimensions to [C, H, W]
        return image_tensor
                    

    def predict(self, file_path: str):
        image_tensor = self.load_image(file_path).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model([image_tensor])  # Ensure to pass a list of tensors
        return predictions[0]

    def show_image(self, file_path: str, predictions):
        image = Image.open(file_path)
        draw = ImageDraw.Draw(image)
        print(predictions)
        for box, label in zip(predictions['boxes'], predictions['labels']):
            color = 'red' if label == 1 else 'blue'
            draw.rectangle(box.tolist(), outline=color, width=2)
            print([box, label])
        image.show()

def main():
    model_path = "model.pth"
    predictor = FasterRCNNPredict(model_path)

    for i in range(11, 51):
        image_path = f"TrainingImages\\image{i}.tif"
        predictions = predictor.predict(image_path)
        predictor.show_image(image_path, predictions)

if __name__ == "__main__":
    main()
