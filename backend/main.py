from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import time

VIDEO_PATH = "testing.mp4"

MODEL_PATH = "yolov8n_openvino_model"

# First run: download yolov8n.pt and export to OpenVINO
if not Path(MODEL_PATH).exists():
    print("OpenVINO model not found. Exporting from yolov8n.pt (first run only)...")
    YOLO("yolov8n.pt").export(format="openvino", dynamic=False)

QUEUE_ROI = np.array([
    (110, 250),
    (670, 230),
    (655, 375),
    (95, 395)
], dtype=np.int32)

# --- count stability ---
MIN_HITS_TO_CONFIRM = 5   # frames a track must persist in ROI before counted (~0.2 s)
GRACE_FRAMES = 15         # keep counting a track this long after a missed detection (~0.5 s)
DUP_IOU_THRESH = 0.80     # two ROI boxes overlapping this much = same person

model = YOLO(MODEL_PATH)

print("Model loaded.")

cap = cv2.VideoCapture(VIDEO_PATH)

fps_video = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

OUTPUT_PATH = "queuesense_roi.mp4"

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (width, height)
)

frame_count = 0
start_time = time.time()

frame_idx = 0
track_hits = {}        # track_id -> frames seen inside ROI
track_last_seen = {}   # track_id -> last frame_idx seen inside ROI
confirmed = set()      # track_ids currently counted in the queue
count_history = []


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack_custom.yaml",
        classes=[0],          # person only
        conf=0.25,
        device="CPU",
        verbose=False
    )

    result = results[0]

    cv2.polylines(
        frame,
        [QUEUE_ROI],
        isClosed=True,
        color=(255, 0, 0),
        thickness=3
    )

    frame_idx += 1

    roi_tracks = []   # (track_id, box, anchor) with feet inside ROI this frame

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = box.astype(int)

            # Bottom-center of bounding box
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # Is person inside queue ROI?
            inside = cv2.pointPolygonTest(
                QUEUE_ROI,
                (cx, cy),
                False
            )

            if inside >= 0:
                roi_tracks.append((int(track_id), (x1, y1, x2, y2), (cx, cy)))

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Same person picked up as two tracks: prefer already-confirmed
        # (stable) tracks, then bigger boxes; drop the rest
        roi_tracks.sort(
            key=lambda t: (
                t[0] in confirmed,
                (t[1][2] - t[1][0]) * (t[1][3] - t[1][1])
            ),
            reverse=True
        )

        kept = []

        for track in roi_tracks:

            if any(iou(track[1], k[1]) > DUP_IOU_THRESH for k in kept):
                continue

            kept.append(track)

    else:
        kept = []

    for track_id, box, (cx, cy) in kept:

        track_hits[track_id] = track_hits.get(track_id, 0) + 1
        track_last_seen[track_id] = frame_idx

        # Only count tracks that have proven stable
        if track_hits[track_id] >= MIN_HITS_TO_CONFIRM:
            confirmed.add(track_id)

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 255, 0),
            -1
        )

    # Keep counting tracks that briefly drop out of detections,
    # expire them after the grace period
    for track_id in list(track_last_seen):

        if frame_idx - track_last_seen[track_id] > GRACE_FRAMES:
            track_last_seen.pop(track_id)
            track_hits.pop(track_id)
            confirmed.discard(track_id)

    queue_count = len(confirmed)

    count_history.append(queue_count)

    frame_count += 1

    elapsed = time.time() - start_time

    current_fps = frame_count / elapsed

    # Bottom-center overlay box
    box_w, box_h, margin = 400, 105, 20
    bx = (width - box_w) // 2
    by = height - box_h - margin

    cv2.rectangle(
        frame,
        (bx, by),
        (bx + box_w, by + box_h),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"QUEUE: {queue_count}",
        (bx + 20, by + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame,
        f"FPS: {current_fps:.1f}",
        (bx + 20, by + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    writer.write(frame)

cap.release()
writer.release()

count_changes = sum(1 for a, b in zip(count_history, count_history[1:]) if a != b)

print()
print("========================================")
print("        QUEUESENSE ROI TEST")
print("========================================")
print(f"Frames processed : {frame_count}")
print(f"Average FPS      : {current_fps:.2f}")
print(f"Count changes    : {count_changes} frame-to-frame (lower = stabler)")
print(f"Output           : {OUTPUT_PATH}")
print("========================================")
