# 📷 Object Scanner

A real-time webcam object scanner. It detects everyday objects with **YOLOv8**, draws labelled boxes on the live video, and looks up a one-sentence **Wikipedia** summary for each new object it sees, all without slowing down the video feed.



## Features

- **Real-time detection** using the lightweight YOLOv8 nano model (`yolov8n.pt`) on 80 COCO object classes
- **Live overlay** with bounding boxes, class labels, and confidence scores
- **Background Wikipedia lookups**: summaries are fetched in worker threads, so the video never freezes while waiting on the network
- **Smart caching**: each object type is looked up only once per session
- **Disambiguation fixes**: a small title map makes ambiguous labels resolve correctly (e.g. `mouse` → *Computer mouse*, `remote` → *Remote control*, `tie` → *Necktie*)
- **Clean console output**: each object is announced once, not on every frame
- **Safe shutdown**: the camera and windows are always released, even on errors

## How it works

```
Webcam frame ──► YOLOv8 inference ──► boxes + labels drawn on frame
                                            │
                                    new label seen?
                                            │
                         background thread ─┴─► Wikipedia summary ──► cache ──► console
```

1. OpenCV grabs a frame from the webcam.
2. YOLOv8 detects objects (confidence threshold `0.4`).
3. Each detection is drawn on the frame.
4. The first time a label appears, a daemon thread fetches its Wikipedia summary and stores it in a cache (a placeholder `"Loading..."` prevents duplicate threads).
5. Once the summary is ready, it is printed a single time.

## Requirements

- Python 3.8+
- A working webcam
- Internet connection (for the first download of the YOLO weights and for Wikipedia lookups)

## Installation

```bash
git clone https://github.com/sankhasuvraghosh/real-time-object-scanner.git
cd real-time-object-scanner

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install ultralytics opencv-python wikipedia
```

`ultralytics` installs PyTorch automatically. The `yolov8n.pt` weights are downloaded on first run.

## Usage

```bash
python object_wiki.py
```

- Point your webcam at objects; boxes and labels appear in the **Object Scanner** window.
- Object summaries print to the terminal the first time each object is recognized.
- Press **`q`** in the video window to quit.

Example console output:

```
✅ Webcam started. Press 'q' to quit.
Detected: laptop → A laptop computer, also known as a notebook computer, is a small, portable personal computer...
Detected: cell phone → A mobile phone is a portable telephone that can make and receive calls...
```

## Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| Model | `YOLO("yolov8n.pt")` | `yolov8n.pt` | Swap for `yolov8s.pt`, `yolov8m.pt`, etc. for higher accuracy at lower speed |
| Confidence threshold | `model(frame, conf=0.4)` | `0.4` | Raise to reduce false positives, lower to detect more |
| Camera index | `cv2.VideoCapture(0)` | `0` | Try `1` or `2` for an external webcam |
| Wikipedia title fixes | `WIKI_TITLES` dict | a few common labels | Add entries for any label that lands on the wrong page |
| Summary length | `sentences=1` | `1` | Increase for longer summaries |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not access webcam` | Close other apps using the camera, or change the camera index. On macOS/Linux, check camera permissions. |
| Video is laggy on CPU | Use `yolov8n.pt`, run inference on every 2nd or 3rd frame, or reduce the frame size with `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`. |
| Wrong Wikipedia page for an object | Add the label and the correct article title to `WIKI_TITLES`. |
| `Too many meanings: [...]` | The label is ambiguous; add a `WIKI_TITLES` entry to pin the right article. |
| `Error:` messages from Wikipedia | Usually a network issue or rate limiting; the lookup for that label is cached, so restart to retry. |

## Project structure

```
.
├── object_wiki.py      # main script
└── README.md
```

## Ideas for future work

- Show the Wikipedia summary directly on the video frame (info panel)
- Object tracking with `model.track(frame, persist=True)` for stable IDs
- Retry failed Wikipedia lookups instead of caching the error
- Text-to-speech announcements of detected objects
- Custom-trained model for domain-specific objects

## Built with

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [wikipedia](https://pypi.org/project/wikipedia/) Python package

