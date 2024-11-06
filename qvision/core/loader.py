import torch
import torchvision.models as models
import tensorflow as tf
import onnx
import onnxruntime as ort

class ModelLoader:

    def __init__(self):
        pass

    def load_pytorch_model(self, model_name = 'resnet18', pretrained = true, model_path = None):
        