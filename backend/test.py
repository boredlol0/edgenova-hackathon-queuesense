from ultralytics import YOLO
import cv2
import time

# 1. Export the model to OpenVINO once
base_model = YOLO("yolov8n.pt")
# base_model.export(format="openvino", dynamic=False)

# 2. Load the exported OpenVINO model
# This folder is created by the export step above
model = YOLO("yolov8n_openvino_model/")

video_path = "./testing.mp4"
devices_to_test = ["CPU"]
results_summary = {}

# 3. Benchmark each device
for device_name in devices_to_test:
    print(f"Starting benchmark for {device_name}...")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference using OpenVINO on the specific device
        results = model.track(
            frame,
            classes=[0],       # person only
            tracker="bytetrack.yaml",
            persist=True,
            conf=0.35,
            device=device_name,
            verbose=False
        )

        frame_count += 1

    cap.release()

    elapsed = time.time() - start
    fps = frame_count / elapsed if elapsed > 0 else 0
    results_summary[device_name] = (frame_count, elapsed, fps)
    
    print(f"{device_name} completed.\n")

# 4. Print final benchmark results
print("=" * 40)
print("        YOLO BENCHMARK RESULTS        ")
print("=" * 40)
for dev, (frames, time_taken, fps_val) in results_summary.items():
    print(f"{dev:<5} | Frames: {frames:<4} | Time: {time_taken:>5.2f}s | FPS: {fps_val:>6.2f}")
print("=" * 40)
