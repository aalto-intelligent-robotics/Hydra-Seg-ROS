from ultralytics import YOLO
import argparse
from pathlib import Path

HOME = Path.home()


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default=HOME / "models/yolo/yolo11l-seg.pt")
args = parser.parse_args()

model = YOLO(args.input)
model.export(
    format="engine",
    imgsz=800,
    # device=0,
    # int8=False,
    # half=False,
    dynamic=True,
    # simplify=True,
)
