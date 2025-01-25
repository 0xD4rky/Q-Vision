from ultralytics import YOLO
import argparse
import pathlib

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='validation model')
    parser.add_argument('--model', type=str, default='model/yolov8n.engine', required=True, help='Path to the .pt')
    parser.add_argument('--q', type=str, default='fp16', required=False, help='[fp16, int8]')
    parser.add_argument('--data', type=str, default='coco.yaml', required=True, help='Dataset')
    args = parser.parse_args()
