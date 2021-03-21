from .nanodet import NanodetLightning


def create_model(model_cfg:dict):
    name = model_cfg['name']
    if name == 'YOLO':
        pass
    elif name == 'NanoDet':
        return NanodetLightning(model_cfg['arch'])
