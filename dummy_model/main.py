import json


def init_context(context):
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run Dummy model")

    # just for debugging
    debug_results = [{
        "confidence": str(1.0),
        "label": "ball",
        "points": [float(0.1), float(0.2), float(0.3), float(0.4)],
        "type": "rectangle",
    }]
    return context.Response(body=json.dumps(debug_results), headers={},
            content_type='application/json', status_code=200)

