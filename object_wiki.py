import cv2
import torch
import wikipedia
import pandas as pd
import threading

# -------------------------
# Load YOLOv8 model
# -------------------------
try:
    # Install YOLOv8 if not already: pip install ultralytics
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # Lightweight YOLOv8 nano model
except Exception as e:
    print("❌ Failed to load YOLOv8 model. Install with 'pip install ultralytics'.")
    raise e

# -------------------------
# Start webcam
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not access webcam.")
print("✅ Webcam started. Press 'q' to quit.")

# -------------------------
# Cache & thread helpers
# -------------------------
wiki_cache = {}         # label -> summary
printed_labels = set()  # avoid multiple thread launches

def fetch_wiki(label):
    """Fetch Wikipedia summary for a label (runs in background thread)."""
    try:
        wiki_cache[label] = wikipedia.summary(label, sentences=1)
    except wikipedia.exceptions.DisambiguationError as e:
        wiki_cache[label] = f"Too many meanings: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        wiki_cache[label] = "Page not found."
    except Exception as e:
        wiki_cache[label] = f"Error: {str(e)}"

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    # Run YOLO detection
    results = model(frame)[0]  # first result

    # Convert detections to DataFrame
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = results.names[cls]
            detections.append({
                "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                "confidence": conf, "class": cls, "name": name
            })
    detections = pd.DataFrame(detections)

    # Process detections
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        label = row['name']

        # Draw rectangle + label in black
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        y_text = max(y1 - 10, 20)
        cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Launch Wikipedia fetch in background if not cached
        if label not in wiki_cache and label not in printed_labels:
            printed_labels.add(label)
            threading.Thread(target=fetch_wiki, args=(label,), daemon=True).start()

        # Print cached info if available
        if label in wiki_cache:
            print(f"Detected: {label} → {wiki_cache[label]}")

    # Show video
    cv2.imshow("📷 Object Scanner", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()