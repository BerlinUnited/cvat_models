import os
from copy import copy
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import numpy as np
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from pathlib import Path

class ModelHandler:
    def __init__(self):
        # Setup device
        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.WEIGHTS = "model_final.pth"
        print("IS MODEL THERE?: ", Path("model_final.pth").exists())


        self.img_transformer = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

        self.model = build_model(cfg)  # returns a torch.nn.Module
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS


    def infer(self, image):
        height, width = image.shape[:2]
        image = np.transpose(np.array(image), (2, 0, 1))  # change from HWC to CHW

        img_tensor = torch.from_numpy(image)
        self.model.eval()
        image_list = [{"image": img_tensor, "height": height, "width": width}]
        with torch.no_grad():
            outputs = self.model(image_list)[0]

        print(outputs)
        return outputs