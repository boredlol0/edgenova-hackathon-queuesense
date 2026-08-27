from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import time

VIDEO_PATH = "test1.mp4"

# --- HF crowd model: AmineSam/irail-crowd-counting-yolov8n (YOLOv8n head detector, imgsz=832) ---
HF_REPO = "AmineSam/irail-crowd-counting-yolov8n"
HF_FILENAME = "best.pt"
CROWD_PT_CACHE = Path("irail_crowd_best.pt")  # local copy after hf download
# Export creates {stem}_openvino_model, i.e. irail_crowd_best_openvino_model
MODEL_PATH = "irail_crowd_best_openvino_model"  # OpenVINO export dir (must match YOLO export name)
# fallback for legacy name if user had old export
_LEGACY_MODEL_PATH = "irail_crowd_openvino_model"

def get_crowd_pt():
    """Download best.pt from HF (case-sensitive AmineSam) to local cache if needed."""
    if CROWD_PT_CACHE.exists():
        return str(CROWD_PT_CACHE)
    print(f"Downloading {HF_REPO}/{HF_FILENAME} from Hugging Face...")
    try:
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
        # copy to local cache for stable export path
        import shutil
        shutil.copy(tmp, CROWD_PT_CACHE)
        print(f"Saved to {CROWD_PT_CACHE}")
        return str(CROWD_PT_CACHE)
    except Exception as e:
        print(f"[error] HF download failed: {e}")
        print("Try: huggingface-cli login, or check network/token for gated repo")
        raise

# Ensure PT exists before export
crowd_pt = get_crowd_pt()

# First run: export crowd model to OpenVINO (CPU) - separate from yolov8n_openvino_model
# Handle legacy path irail_crowd_openvino_model vs correct irail_crowd_best_openvino_model
if Path(_LEGACY_MODEL_PATH).exists() and not Path(MODEL_PATH).exists():
    MODEL_PATH = _LEGACY_MODEL_PATH
if not Path(MODEL_PATH).exists():
    print(f"OpenVINO model not found at {MODEL_PATH}. Exporting {crowd_pt} (first run, ~1-2 min)...")
    exported = YOLO(crowd_pt).export(format="openvino", dynamic=False)
    # YOLO.export returns exported dir path - use it if MODEL_PATH mismatched
    if exported and Path(str(exported)).exists():
        MODEL_PATH = str(exported)
    print(f"Export done: {MODEL_PATH}")

# ROI for HEAD-based counting - same lane but top edge raised ~100px
# so heads (not feet) just inside doorway still count. Tune live via clicks.
# (portrait video 1080x1920)
# QUEUE_ROI = np.array([
#     (445, 620),
#     (685, 620),
#     (860, 1385),
#     (340, 1385)
# ], dtype=np.int32)
QUEUE_ROI = np.array([(442, 759), (748, 747), (924, 1881), (280, 1905), (446, 756)], dtype=np.int32)

# --- live cursor / ROI calibration ---
# mouse_pos updated by mouse callback, shown at cursor
mouse_pos = [0, 0]
clicked_points = []  # left-clicks collected for quick ROI copy-paste

def on_mouse(event, x, y, flags, param):
    global mouse_pos, clicked_points, QUEUE_ROI
    mouse_pos[0], mouse_pos[1] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"[ROI] Click {len(clicked_points)}: ({x}, {y})  |  Current polygon: {clicked_points}")
        if len(clicked_points) == 4:
            QUEUE_ROI = np.array(clicked_points, dtype=np.int32)
            print(f"[ROI] Updated QUEUE_ROI to {clicked_points}")
            print("      Paste this into QUEUE_ROI or press 'c' to confirm / 'r' to reset")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicked_points:
            clicked_points.pop()
            print(f"[ROI] Undo, remaining: {clicked_points}")

# --- count stability (tuned for crowd - lower thresholds, longer grace) ---
MIN_HITS_TO_CONFIRM = 2   # 5->2: occluded heads flicker, confirm faster
GRACE_FRAMES = 30         # 15->30: keep counting through short occlusions
DUP_IOU_THRESH = 0.45     # 0.80->0.45: overlap in dense queue ~0.3-0.5, dedup duplicates

model = YOLO(MODEL_PATH)

print(f"Model loaded: {HF_REPO} -> {MODEL_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

fps_video = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- output modes ---
SHOW_LIVE = True   # stream to window as frames are processed (instant feedback)
SAVE_OUTPUT = True  # keep mp4 write; set False to skip disk I/O on long videos
OUTPUT_PATH = "queuesense_roi.mp4"

writer = None
if SAVE_OUTPUT:
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_video,
        (width, height)
    )

if SHOW_LIVE:
    try:
        # Window sized to video dims, aspect preserved (letterboxed, not stretched).
        # WINDOW_KEEPRATIO ensures resize keeps aspect; image is never stretched to fill.
        # For strict 1:1 non-resizable, use WINDOW_AUTOSIZE instead.
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_KEEPRATIO"):
            flags |= cv2.WINDOW_KEEPRATIO
        if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
            flags |= cv2.WINDOW_GUI_EXPANDED
        cv2.namedWindow("QueueSense - Live", flags)
        # Initial window size = video size (1:1). User can resize, but ratio is kept.
        cv2.resizeWindow("QueueSense - Live", width, height)
        cv2.setMouseCallback("QueueSense - Live", on_mouse)
    except cv2.error as e:
        print(f"[warn] Live window unavailable ({e}) - continuing without preview")
        SHOW_LIVE = False

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

    # Crowd model baseline: imgsz=832, conf=0.25, iou=0.75, max_det=300 per model card
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack_custom.yaml",
        classes=[0],          # head only (single class model)
        conf=0.25,
        iou=0.75,
        imgsz=832,
        max_det=300,
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

    roi_tracks = []   # (track_id, box, anchor) with HEAD inside ROI this frame

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = box.astype(int)

            # Head-center of bounding box (top-center, ~12% down from top)
            # More stable than feet when feet are occluded in crowds
            h = y2 - y1
            cx = int((x1 + x2) / 2)
            cy = int(y1 + h * 0.12)

            # Is person's HEAD inside queue ROI?
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

        # --- head box (replaces green dot) ---
        bx1, by1, bx2, by2 = box  # original person box
        w = bx2 - bx1
        h = by2 - by1
        head_w = int(w * 0.44)
        head_h = int(h * 0.30)
        hx1 = int(cx - head_w // 2)
        hx2 = int(cx + head_w //2)
        hy1 = int(by1)
        hy2 = int(by1 + head_h)
        # clamp to frame
        hx1 = max(0, hx1); hy1 = max(0, hy1)
        hx2 = min(width - 1, hx2); hy2 = min(height - 1, hy2)
        is_confirmed = track_id in confirmed
        head_color = (0, 255, 0) if is_confirmed else (0, 255, 255)  # green=counted, yellow=pending
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), head_color, 2)
        # small center dot for anchor
        cv2.circle(frame, (cx, cy), 2, head_color, -1)
        # head label
        cv2.putText(frame, f"H{track_id}", (hx1, hy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, head_color, 1)

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

    # Bottom-center overlay box (expanded for debug)
    box_w, box_h, margin = 420, 130, 20
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
        (bx + 20, by + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # debug breakdown: helps tune crowd (raw heads inside, after dedup, confirmed)
    cv2.putText(
        frame,
        f"raw:{len(roi_tracks)} kept:{len(kept)} conf:{queue_count}",
        (bx + 20, by + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1
    )

    # --- cursor coordinate overlay (always on live frame, before write/imshow) ---
    if SHOW_LIVE:
        mx, my = mouse_pos
        # crosshair at cursor
        if 0 <= mx < width and 0 <= my < height:
            cv2.drawMarker(frame, (mx, my), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
            # label next to cursor with background for readability
            label = f"({mx}, {my})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx = min(mx + 12, width - tw - 8)
            ly = max(my - 12, th + 8)
            cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        # help bar at top
        help_txt = "Hover: coords | Left-click: add ROI point (4=update) | Right-click: undo | r: reset | q/ESC: quit"
        (htw, hth), _ = cv2.getTextSize(help_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (4, 4), (htw + 12, hth + 12), (0, 0, 0), -1)
        cv2.putText(frame, help_txt, (8, hth + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        # draw in-progress ROI clicks
        for i, pt in enumerate(clicked_points):
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{i+1}", (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(clicked_points) > 1:
            cv2.polylines(frame, [np.array(clicked_points, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)

    if SAVE_OUTPUT and writer is not None:
        writer.write(frame)

    # --- livestream to window (no buffering) ---
    if SHOW_LIVE:
        cv2.imshow("QueueSense - Live", frame)
        # waitKey(1) pumps window events; 'q' or ESC to quit early (saves time on long videos)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print(f"\nInterrupted by user at frame {frame_count}")
            break
        elif key == ord('r'):
            clicked_points.clear()
            print("[ROI] Reset clicked points")
        elif key == ord('c') and len(clicked_points) == 4:
            QUEUE_ROI = np.array(clicked_points, dtype=np.int32)
            print(f"[ROI] Confirmed QUEUE_ROI = {clicked_points}")
        # if window was closed by user, exit cleanly
        try:
            if cv2.getWindowProperty("QueueSense - Live", cv2.WND_PROP_VISIBLE) < 1:
                print(f"\nWindow closed at frame {frame_count}")
                break
        except cv2.error:
            break

cap.release()
if writer is not None:
    writer.release()
if SHOW_LIVE:
    cv2.destroyAllWindows()

count_changes = sum(1 for a, b in zip(count_history, count_history[1:]) if a != b)

print()
print("========================================")
print("        QUEUESENSE ROI TEST (CROWD)")
print("========================================")
print(f"Model            : {HF_REPO}")
print(f"Frames processed : {frame_count}")
print(f"Average FPS      : {current_fps:.2f}")
print(f"Count changes    : {count_changes} frame-to-frame (lower = stabler)")
if SAVE_OUTPUT:
    print(f"Output           : {OUTPUT_PATH}")
else:
    print(f"Output           : (live only, not saved - set SAVE_OUTPUT=True to write mp4)")
print("========================================")
