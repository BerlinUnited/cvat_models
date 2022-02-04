import json
import base64
from PIL import Image
import io
from model_handler import ModelHandler
import cv2
import torch


def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    model = ModelHandler()
    setattr(context.user_data, 'model', model)

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run Detectron2 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
    image = Image.open(buf)

    image.save('test.png')
    cv_img = cv2.imread('test.png')

    outputs = context.user_data.model.infer(cv_img)

    # return an empty list if no bounding boxes were found
    if len(outputs['instances']) == 0:
        return context.Response(body=json.dumps(list()), headers={},
        content_type='application/json', status_code=200)
    else:
        # FIXME assumes that only one ball per image exists
        argmax = torch.argmax(outputs['instances'].get('scores')).item()
        max_conf = torch.max(outputs['instances'].get('scores')).item()
        if max_conf > 0.7:
            best_box = outputs['instances'].get('pred_boxes')[argmax]
            best_box = best_box[0].tensor.numpy()[0]
            best_box_list = [float(best_box[0]), float(best_box[1]), float(best_box[2]), float(best_box[3])]

            results = list()
            results.append({
                        "confidence": str(max_conf),
                        "label": "ball",
                        "points": best_box_list,
                        "type": "rectangle",
                    })
        else:
            results = list()
        return context.Response(body=json.dumps(results), headers={},
            content_type='application/json', status_code=200)