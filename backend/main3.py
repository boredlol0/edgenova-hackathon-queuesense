from pathlib import Path
import sys
import cv2
import numpy as np
import time
import json
import argparse
import torch
from PIL import Image
import torchvision.transforms as standard_transforms

# --- PET repo setup ---
PET_ROOT = Path("PET")
if not PET_ROOT.exists():
    print("Cloning PET repo (cxliu0/PET)...")
    import subprocess
    subprocess.check_call(["git", "clone", "https://github.com/cxliu0/PET.git", str(PET_ROOT)])
    print("PET cloned.")

# Ensure PET is importable
sys.path.insert(0, str(PET_ROOT.resolve()))
import util.misc as utils
from models import build_model

VIDEO_PATH = "test1.mp4"

# --- HF PET crowd model: Awiros/crowd-counting-and-localization (PET Finetuned, point detection) ---
HF_REPO = "Awiros/crowd-counting-and-localization"
HF_WEIGHT = "PET_Finetuned.safetensors"
LOCAL_WEIGHT = Path("PET_Finetuned.safetensors")
MODEL_PATH_HINT = "PET_Finetuned"  # not OpenVINO, PET uses PyTorch safetensors

def get_pet_weights():
    if LOCAL_WEIGHT.exists():
        return str(LOCAL_WEIGHT)
    print(f"Downloading {HF_REPO}/{HF_WEIGHT} from Hugging Face...")
    from huggingface_hub import hf_hub_download
    import shutil
    tmp = hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHT)
    shutil.copy(tmp, LOCAL_WEIGHT)
    print(f"Saved to {LOCAL_WEIGHT} ({LOCAL_WEIGHT.stat().st_size/1e6:.1f} MB)")
    return str(LOCAL_WEIGHT)

weights_path = get_pet_weights()

# ROI for HEAD-based counting (points) - same lane, for PET points (head centers)
QUEUE_ROI = np.array([(442, 759), (748, 747), (924, 1881), (280, 1905), (446, 756)], dtype=np.int32)

# --- live cursor / ROI calibration ---
mouse_pos = [0, 0]
clicked_points = []

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

# --- count stability for point model (no IDs) -> temporal smoothing ---
SMOOTH_WINDOW = 15  # median over last 15 frames (~0.6s @25fps)

# --- PET helpers (from HF test.py, trimmed) ---
PET_TRANSFORM = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def resize_for_eval(frame_rgb, upper_bound):
    h, w = frame_rgb.shape[:2]
    max_size = max(h, w)
    if upper_bound != -1 and max_size > upper_bound:
        scale = float(upper_bound) / float(max_size)
    elif max_size > 2560:
        scale = 2560.0 / float(max_size)
    else:
        scale = 1.0
    if scale == 1.0:
        return frame_rgb, scale
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale

def _load_state_dict(weight_path: Path):
    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors
        return load_safetensors(str(weight_path), device="cpu")
    ckpt = torch.load(str(weight_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt

def build_pet_args():
    # mirrors HF test.py defaults for finetuned PET
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)
    parser.add_argument('--dataset_file', default='SHA')
    parser.add_argument('--data_path', default='./data/ShanghaiTech/PartA', type=str)
    parser.add_argument('--upper_bound', default=-1, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    # parse empty to get defaults
    args = parser.parse_args([])
    # ensure CPU on this machine (model supports CPU, not GPU-only)
    args.device = "cpu"
    return args

def resolve_device(device_str: str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)

@torch.no_grad()
def infer_pet_points(model, frame_bgr, device, upper_bound=-1):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb, scale = resize_for_eval(frame_rgb, upper_bound)
    resized_h, resized_w = resized_rgb.shape[:2]
    img = Image.fromarray(resized_rgb)
    img = PET_TRANSFORM(img)
    samples = utils.nested_tensor_from_tensor_list([img]).to(device)
    img_h, img_w = samples.tensors.shape[-2:]
    outputs = model(samples, test=True)
    outputs_points = outputs["pred_points"]
    if outputs_points.dim() == 3:
        outputs_points = outputs_points[0]
    pred_points = outputs_points.detach().cpu().numpy()
    if pred_points.size == 0:
        return np.zeros((0,2), dtype=np.float32), scale
    pred_points[:,0] *= float(img_h)
    pred_points[:,1] *= float(img_w)
    pred_points[:,0] = np.clip(pred_points[:,0], 0.0, float(resized_h-1))
    pred_points[:,1] = np.clip(pred_points[:,1], 0.0, float(resized_w-1))
    if scale != 1.0:
        pred_points = pred_points / float(scale)
    orig_h, orig_w = frame_bgr.shape[:2]
    pred_points[:,0] = np.clip(pred_points[:,0], 0.0, float(orig_h-1))
    pred_points[:,1] = np.clip(pred_points[:,1], 0.0, float(orig_w-1))
    points_xy = np.stack([pred_points[:,1], pred_points[:,0]], axis=1)  # x,y
    return points_xy, scale

# --- Ensure VGG pretrained weights (PET backbone expects ./pretrained/vgg16_bn-6c64b313.pth) ---
# No, it doesn't mean GPU-only. The missing 'device' arg is now fixed, but PET also needs VGG16_BN ImageNet weights
# which are not in the HF release. Download once to both possible cwd locations.
import urllib.request
vgg_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
for p in [Path("PET/pretrained/vgg16_bn-6c64b313.pth"), Path("pretrained/vgg16_bn-6c64b313.pth")]:
    if not p.exists():
        print(f"Downloading VGG16_BN pretrained for PET backbone to {p} ...")
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            # download with progress
            def _report(block_num, block_size, total_size):
                if block_num % 100 == 0:
                    print(f"  {block_num*block_size/1e6:.1f}/{total_size/1e6:.1f} MB")
            urllib.request.urlretrieve(vgg_url, str(p), reporthook=_report)
            print(f"Saved VGG weights to {p}")
        except Exception as e:
            print(f"Failed to download VGG weights: {e}")
            # copy from other location if available
            alt = Path("PET/pretrained/vgg16_bn-6c64b313.pth") if p == Path("pretrained/vgg16_bn-6c64b313.pth") else Path("pretrained/vgg16_bn-6c64b313.pth")
            if alt.exists():
                import shutil
                shutil.copy(alt, p)
        # ensure both locations have file
        other = Path("pretrained/vgg16_bn-6c64b313.pth") if p == Path("PET/pretrained/vgg16_bn-6c64b313.pth") else Path("PET/pretrained/vgg16_bn-6c64b313.pth")
        if not other.exists() and p.exists():
            import shutil
            other.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, other)

# --- Load PET model (CPU supported, not GPU-only) ---
device = resolve_device("cpu")
pet_args = build_pet_args()
# FIX: PET's build_pet requires args.device (was missing -> AttributeError)
model, _ = build_model(pet_args)
model.to(device)
model.eval()
print(f"Loading PET weights: {weights_path}")
state_dict = _load_state_dict(Path(weights_path))
# PET checkpoint may have extra keys: load strictly
model.load_state_dict(state_dict, strict=True)
print(f"Model loaded: {HF_REPO} -> {weights_path} (device={device})")

cap = cv2.VideoCapture(VIDEO_PATH)
fps_video = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

SHOW_LIVE = True
SAVE_OUTPUT = True
OUTPUT_PATH = "queuesense_roi_pet.mp4"

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
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_KEEPRATIO"):
            flags |= cv2.WINDOW_KEEPRATIO
        if hasattr(cv2, "WINDOW_GUI_EXPANDED"):
            flags |= cv2.WINDOW_GUI_EXPANDED
        cv2.namedWindow("QueueSense - PET Live", flags)
        cv2.resizeWindow("QueueSense - PET Live", width, height)
        cv2.setMouseCallback("QueueSense - PET Live", on_mouse)
    except cv2.error as e:
        print(f"[warn] Live window unavailable ({e}) - continuing without preview")
        SHOW_LIVE = False

frame_count = 0
start_time = time.time()
count_history_raw = []  # per-frame raw inside ROI
count_history_smooth = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # PET inference (point detection)
    points_xy, scale = infer_pet_points(model, frame, device, upper_bound=-1)
    # Filter points inside ROI
    inside_pts = []
    outside_pts = []
    for (x,y) in points_xy:
        # cv2.pointPolygonTest expects (x,y)
        inside = cv2.pointPolygonTest(QUEUE_ROI, (float(x), float(y)), False) >= 0
        if inside:
            inside_pts.append((int(x), int(y)))
        else:
            outside_pts.append((int(x), int(y)))

    raw_inside = len(inside_pts)
    count_history_raw.append(raw_inside)
    # keep window
    if len(count_history_raw) > SMOOTH_WINDOW * 4:
        count_history_raw = count_history_raw[-(SMOOTH_WINDOW*4):]
    # smoothed queue count = median of last SMOOTH_WINDOW frames (robust to flicker)
    window = count_history_raw[-SMOOTH_WINDOW:] if len(count_history_raw) >= SMOOTH_WINDOW else count_history_raw
    if window:
        queue_count = int(np.median(window))
    else:
        queue_count = 0
    count_history_smooth.append(queue_count)

    frame_count += 1
    elapsed = time.time() - start_time
    current_fps = frame_count / elapsed if elapsed>0 else 0

    # Draw ROI
    cv2.polylines(frame, [QUEUE_ROI], isClosed=True, color=(255, 0, 0), thickness=3)

    # Draw PET points: head boxes around points (user requested boxes, not just dots)
    # Inside ROI: green head boxes (12x12 square or 10px box)
    # Outside ROI: small gray dots for context
    for (x,y) in outside_pts:
        cv2.circle(frame, (x,y), 2, (100,100,100), -1)
    for (x,y) in inside_pts:
        # head box 14x14 centered at point (approximate head size, consistent regardless of scale)
        box = 14
        hx1, hy1 = int(x - box//2), int(y - box//2)
        hx2, hy2 = int(x + box//2), int(y + box//2)
        # head box green, with label
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
        cv2.circle(frame, (x,y), 1, (0,255,0), -1)

    # Bottom overlay (queue + FPS + raw/smooth)
    box_w, box_h, margin = 460, 140, 20
    bx = (width - box_w)//2
    by = height - box_h - margin
    cv2.rectangle(frame, (bx, by), (bx+box_w, by+box_h), (0,0,0), -1)
    cv2.putText(frame, f"QUEUE: {queue_count}", (bx+20, by+38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
    cv2.putText(frame, f"FPS: {current_fps:.1f} (PET)", (bx+20, by+72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(frame, f"raw:{raw_inside} smooth:{queue_count} total_pts:{len(points_xy)}", (bx+20, by+100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    cv2.putText(frame, f"PET {HF_REPO}", (bx+20, by+125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1)

    # --- cursor overlay ---
    if SHOW_LIVE:
        mx,my = mouse_pos
        if 0 <= mx < width and 0 <= my < height:
            cv2.drawMarker(frame, (mx,my), (0,255,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
            label = f"({mx},{my})"
            (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx = min(mx+12, width - tw -8)
            ly = max(my-12, th+8)
            cv2.rectangle(frame, (lx-4, ly-th-4), (lx+tw+4, ly+4), (0,0,0), -1)
            cv2.putText(frame, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
        help_txt = "PET: green=head inside ROI | gray=outside | r:reset c:confirm q:quit"
        (htw,hth), _ = cv2.getTextSize(help_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (4,4), (htw+12, hth+12), (0,0,0), -1)
        cv2.putText(frame, help_txt, (8, hth+8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        for i, pt in enumerate(clicked_points):
            cv2.circle(frame, pt, 5, (0,255,0), -1)
            cv2.putText(frame, f"{i+1}", (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        if len(clicked_points)>1:
            cv2.polylines(frame, [np.array(clicked_points, dtype=np.int32)], isClosed=False, color=(0,255,0), thickness=2)

    if SAVE_OUTPUT and writer is not None:
        writer.write(frame)

    if SHOW_LIVE:
        cv2.imshow("QueueSense - PET Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print(f"\nInterrupted at frame {frame_count}")
            break
        elif key == ord('r'):
            clicked_points.clear()
            print("[ROI] Reset")
        elif key == ord('c') and len(clicked_points)==4:
            QUEUE_ROI = np.array(clicked_points, dtype=np.int32)
            print(f"[ROI] Confirmed {clicked_points}")
        try:
            if cv2.getWindowProperty("QueueSense - PET Live", cv2.WND_PROP_VISIBLE) < 1:
                print(f"\nWindow closed at {frame_count}")
                break
        except cv2.error:
            break

cap.release()
if writer is not None:
    writer.release()
if SHOW_LIVE:
    cv2.destroyAllWindows()

# smooth stability: changes in smoothed count
count_changes = sum(1 for a,b in zip(count_history_smooth, count_history_smooth[1:]) if a!=b)
print()
print("========================================")
print("        QUEUESENSE ROI TEST (PET)")
print("========================================")
print(f"Model            : {HF_REPO}")
print(f"Frames processed : {frame_count}")
print(f"Average FPS      : {current_fps:.2f}")
print(f"Count changes    : {count_changes} (smoothed, lower=stabler)")
print(f"Raw median range : {min(count_history_raw) if count_history_raw else 0} - {max(count_history_raw) if count_history_raw else 0}")
if SAVE_OUTPUT:
    print(f"Output           : {OUTPUT_PATH}")
else:
    print(f"Output           : (live only)")
print("========================================")
