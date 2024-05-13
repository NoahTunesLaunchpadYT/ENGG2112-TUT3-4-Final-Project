from PyQt5.QtCore import QSize, QPointF, QTimer
from PyQt5.QtGui import QColor
from qgis.core import QgsProject, QgsMapSettings, QgsMapRendererParallelJob, QgsRectangle
import os
import json

# Configuration Variables
base_directory = "C:/Users/noahs/OneDrive/Desktop/School/2024 Sem 1/ENGG2112/RCNN Model/TrainingImages/"
extent_offset = 60  # Offset for bounding box size around the point
image_size = 416  # Size of the output image
bbox_size = 52  # Size of the bounding box around the feature

# Ensure the directory exists and is cleared
def setup_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

setup_directory(base_directory)

satellite_layer = QgsProject.instance().mapLayersByName('Google Satellite')[0]
positive_layer = QgsProject.instance().mapLayersByName('warrens and mounds')[0]
negative_layer = QgsProject.instance().mapLayersByName('negatives')[0]
positive_iterator = iter(positive_layer.getFeatures())
negative_iterator = iter(negative_layer.getFeatures())
image_count = 0
total_images = positive_layer.featureCount() + negative_layer.featureCount()

timer = QTimer()
timer.setInterval(50)  # milliseconds
timer.setSingleShot(True)

def process_feature(feature, is_positive):
    global image_count
    image_count += 1

    center = feature.geometry().asPoint()
    extent = QgsRectangle(center.x() - extent_offset, center.y() - extent_offset,
                          center.x() + extent_offset, center.y() + extent_offset)
    output_file = os.path.join(base_directory, f"image{image_count}.tif")
    print(f"Exporting image {image_count}.tif ({image_count}/{total_images})")

    settings = QgsMapSettings()
    settings.setLayers([satellite_layer])
    settings.setBackgroundColor(QColor(255, 255, 255))
    settings.setOutputSize(QSize(image_size, image_size))
    settings.setExtent(extent)

    render = QgsMapRendererParallelJob(settings)
    boxes = []
    labels = []

    def finished():
        img = render.renderedImage()
        img.save(output_file, "TIFF")
        if is_positive:
            for other_feature in positive_layer.getFeatures():
                if extent.contains(other_feature.geometry().asPoint()):
                    rel_x = (other_feature.geometry().asPoint().x() - extent.xMinimum()) / extent.width() * image_size
                    rel_y = (other_feature.geometry().asPoint().y() - extent.yMinimum()) / extent.height() * image_size
                    x_min = int(rel_x - bbox_size / 2)
                    y_min = int(rel_y - bbox_size / 2)
                    x_max = int(rel_x + bbox_size / 2)
                    y_max = int(rel_y + bbox_size / 2)
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(0)
        annotation_data = {"boxes": boxes, "labels": labels}
        with open(os.path.join(base_directory, f"image{image_count}.json"), 'w') as json_file:
            json.dump(annotation_data, json_file, indent=4)
        QTimer.singleShot(50, lambda: process_next_feature(is_positive))

    render.finished.connect(finished)
    render.start()

def process_next_feature(is_positive):
    global positive_iterator, negative_iterator
    try:
        feature = next(positive_iterator if is_positive else negative_iterator)
        process_feature(feature, is_positive)
    except StopIteration:
        print(f'All images from {"positives" if is_positive else "positives and negatives"} exported successfully.')
        if is_positive:
            # If positives are done, start negatives
            process_next_feature(False)

# Start processing positives
process_next_feature(True)
