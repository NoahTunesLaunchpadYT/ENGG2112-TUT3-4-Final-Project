import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import json
import os

def check_raw_data(root, num_samples=10):
    imgs = sorted([f for f in os.listdir(root) if f.endswith('.tif')])
    labels = sorted([f for f in os.listdir(root) if f.endswith('.json')])

    for i in range(min(num_samples, len(imgs))):
        img_path = os.path.join(root, imgs[i])
        label_path = os.path.join(root, labels[i])
        
        # Display the image and its bounding boxes
        img = Image.open(img_path)
        with open(label_path, 'r') as file:
            target = json.load(file)
        
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for box in target['boxes']:
            rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        plt.title(f'Image: {imgs[i]}')
        plt.axis('off')
        plt.show()

        # Optionally, print the image size and bounding box to check consistency
        print(f"Image size: {img.size} (Width x Height)")
        print(f"Bounding Boxes: {target['boxes']}")

# Usage
dataset_root = os.path.join("TrainingImages")
check_raw_data(dataset_root, 20)
