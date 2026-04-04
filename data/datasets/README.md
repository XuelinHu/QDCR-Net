# Datasets

Place raw and processed datasets here.

Suggested layout:

- `URPC2018/`
- `RUOD/`
- `splits/`
- `cache/`

Keep large files out of Git.

Supported training layout for the current minimal pipeline:

- `URPC2018/train/images/*.jpg`
- `URPC2018/train/labels/*.txt`
- `URPC2018/val/images/*.jpg`
- `URPC2018/val/labels/*.txt`

Label files are expected in YOLO format:

- one object per line
- `class_id center_x center_y width height`

The current PyTorch trainer reads up to a fixed number of YOLO boxes per image.
It trains a fixed-query detection model with explicit prediction-target matching, inference-time NMS, and mAP evaluation.
It is still a lightweight research scaffold, not yet a production-grade dense detector.
