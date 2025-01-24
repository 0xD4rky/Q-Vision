from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model to TensorRT')
    parser.add_argument('--model', type=str, default='model/yolov8n.pt', required=True, help='Path to the .pt')
    parser.add_argument('--q', type=str, default='fp16', required=True, help='[fp16, int8]')
    parser.add_argument('--data', type=str, default='coco.yaml', required=True, help='Dataset')
    parser.add_argument('--batch', type=int, default=1, required=False, help='batch size')
    parser.add_argument('--workspace', type=int, default=4, required=False, help='workspace')
    args = parser.parse_args()