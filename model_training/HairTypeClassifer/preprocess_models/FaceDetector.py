import numpy as np

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

MIN_CONFIDENCE = 0.5
# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
# load model
model = YOLO(model_path, task="detect")

# inference
def output_bb(img): 
    output = model(img, verbose=False)

    batch_boxes = []
    for img_results in output:
        cleaned_boxes = [[]]
        max_conf = -1

        if len(img_results.boxes.data) == 0:
            cleaned_boxes[0] = [
                    415, # x1
                    363, # y1
                    615, # x2
                    680, #y2
            ]
        else:
            for box in img_results.boxes.data:
                if box[4] > max_conf:
                    cleaned_boxes[0] = [
                        int(box[0]), # x1
                        int(box[1]), # y1
                        int(box[2]), # x2
                        int(box[3]), # y2
                    ]
                    max_conf = box[4]
                
        batch_boxes.append(cleaned_boxes)

    return np.array(batch_boxes)