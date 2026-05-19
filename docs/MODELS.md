# Models

This repository keeps the model files needed to run the included
residential-parking demo videos.

## Plate Detection

Default:

```text
data/weights/skud.pt
```

`skud.pt` is a YOLOv5 checkpoint from the original project. It is the default
because it gives more stable plate detections on the included parking videos.

Alternative:

```text
data/weights/yolo8s.pt
```

`yolo8s.pt` is loaded through the Ultralytics API and is kept to demonstrate the
same detector wrapper working with newer YOLO checkpoint formats.

## OCR

```text
data/weights/LPRNet.pth
```

LPRNet recognizes the cropped license plate text. The output is normalized
before database lookup, and the pipeline can map one-character OCR variants to a
known whitelist plate.

## Notes

The repository is structured as an end-to-end application demo, not a model
training project. The included scripts make it possible to compare detector
weights on the demo videos, but there is no separate labeled validation set in
this repository.
