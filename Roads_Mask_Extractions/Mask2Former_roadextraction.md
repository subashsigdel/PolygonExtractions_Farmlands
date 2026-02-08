# Mask2Former Road Segmentation — Workflow

## Overview

The `MaskToFormer_road.ipynb` notebook implements a **semantic segmentation** pipeline for road extraction using a fine-tuned **Mask2Former** model with a Swin-Tiny backbone. The model was pre-trained on COCO and fine-tuned on a custom 7-class dataset, with road assigned as **class ID 3**.

---

## Pipeline Architecture

```
Input Image (2048×2048 GeoTIFF)
        │
        ▼
┌───────────────────────────┐
│  Resize to 512×512        │  ← Albumentations
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  HuggingFace Processor    │  ← Normalize + format tensors
│  (Mask2FormerImageProc)   │
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  Mask2Former Inference    │  ← Swin-Tiny + masked attention decoder
│  (7-class semantic seg)   │
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  Post-Process Semantic    │  ← post_process_semantic_segmentation
│  → Per-pixel class map    │
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  Extract Road Class       │  ← class_map == 3 → binary mask
│  (ID = 3)                 │
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  Upscale to 2048×2048     │  ← cv2.INTER_NEAREST
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│  Visualization            │  ← 3-panel: Original | Mask | Overlay
└───────────────────────────┘
```

---

## Step-by-Step Workflow

### Step 1 — Configuration

```python
CONFIG = {
    "MODEL_CHECKPOINT": "facebook/mask2former-swin-tiny-coco-instance",
    "WEIGHTS_PATH": "mask2former.pth",
    "INFER_SIZE": 512,      # Model input resolution
    "ORIG_SIZE": 2048,      # Original tile resolution
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

ROAD_CLASS_ID = 3
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MODEL_CHECKPOINT` | `facebook/mask2former-swin-tiny-coco-instance` | Base architecture from HuggingFace |
| `WEIGHTS_PATH` | `mask2former.pth` | Custom fine-tuned weights (local file) |
| `INFER_SIZE` | 512 | Resolution the model processes (downscaled) |
| `ORIG_SIZE` | 2048 | Resolution of input tiles (upscaled back) |
| `ROAD_CLASS_ID` | 3 | Road class in the 7-class label space |

---

### Step 2 — Model & Processor Loading

```python
# Image processor — handles normalization, NO resizing (we do it manually)
processor = Mask2FormerImageProcessor.from_pretrained(
    CONFIG["MODEL_CHECKPOINT"],
    do_resize=False,        # Resize handled separately via Albumentations
    do_rescale=True,        # Scale pixel values to [0, 1]
    do_normalize=True       # Apply ImageNet normalization
)

# Model — fine-tuned for 7 classes
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    CONFIG["MODEL_CHECKPOINT"],
    num_labels=7,                    # Custom class count
    ignore_mismatched_sizes=True     # Allow head size mismatch from COCO → custom
)

# Load fine-tuned weights
state_dict = torch.load(CONFIG["WEIGHTS_PATH"], map_location="cpu")
model.load_state_dict(state_dict)
model.to(CONFIG["DEVICE"])
model.eval()
```

**Key details:**
- `do_resize=False` — resizing is done manually with Albumentations for explicit control
- `ignore_mismatched_sizes=True` — necessary because the base checkpoint has COCO class count (80+) but our model has 7 classes
- Fine-tuned weights override the entire model state dict, including the segmentation head

---

### Step 3 — Image Preprocessing

```python
resize_to_model = A.Compose([
    A.Resize(512, 512)
])
```

Input images (2048×2048) are resized to 512×512 using Albumentations before being passed to the HuggingFace processor. The processor then applies:
- **Rescaling:** Pixel values from [0, 255] → [0.0, 1.0]
- **Normalization:** ImageNet mean/std normalization

---

### Step 4 — Inference & Road Mask Extraction

The `predict_road_mask_2048()` function handles the full prediction pipeline for a single image:

```python
def predict_road_mask_2048(image_path):
    # 1. Load original 2048×2048 image
    orig_image = np.array(Image.open(image_path).convert("RGB"))
    assert h == 2048 and w == 2048
    
    # 2. Resize to 512×512 for model input
    resized = resize_to_model(image=orig_image)["image"]
    
    # 3. Normalize and format via HuggingFace processor
    inputs = processor(images=resized, return_tensors="pt").to(device)
    
    # 4. Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 5. Post-process to semantic class map (512×512)
    semantic_map = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(512, 512)]
    )[0].cpu().numpy()
    
    # 6. Extract road class as binary mask
    road_mask_small = (semantic_map == ROAD_CLASS_ID).astype(np.uint8)
    
    # 7. Upscale back to 2048×2048
    road_mask = cv2.resize(
        road_mask_small,
        (2048, 2048),
        interpolation=cv2.INTER_NEAREST
    )
    
    return orig_image, road_mask
```

**Processing flow:**

```
2048×2048 RGB image
    │
    ├─ Resize → 512×512
    ├─ Normalize (ImageNet mean/std)
    ├─ Model inference → 7-class logits
    ├─ Argmax → per-pixel class map (512×512)
    ├─ Extract class 3 → binary road mask (512×512)
    └─ Upscale (nearest-neighbor) → binary road mask (2048×2048)
```

**Why nearest-neighbor for upscaling?** The mask is binary (0 or 1). Bilinear or bicubic interpolation would create fractional values at edges, requiring an additional thresholding step. Nearest-neighbor preserves sharp mask boundaries directly.

---

### Step 5 — Visualization

```python
def visualize_matplotlib(orig_image, road_mask):
    overlay = orig_image.copy()
    overlay[road_mask == 1] = [255, 0, 0]   # Red overlay on road pixels
    
    # 3-panel display
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig_image)          # Original satellite image
    axes[1].imshow(road_mask, cmap="gray")  # Binary road mask
    axes[2].imshow(overlay)             # Road overlay in red
```

**Output panels:**

| Panel | Content | Description |
|-------|---------|-------------|
| Left | Original Image | Unmodified 2048×2048 satellite tile |
| Center | Road Mask | Binary mask — white = road, black = non-road |
| Right | Road Overlay | Original image with road pixels colored red |

---

### Step 6 — Batch Folder Inference

```python
def run_folder_inference(folder_path, extensions=(".tif", ".tiff"), pause=False):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
    ])
    
    for idx, image_path in enumerate(image_files, 1):
        orig_image, road_mask = predict_road_mask_2048(image_path)
        visualize_matplotlib(orig_image, road_mask)
```

**Usage:**
```python
run_folder_inference("roads/")
```

Processes all `.tif` and `.tiff` files in the folder sequentially, displaying the 3-panel visualization for each. The optional `pause=True` parameter waits for user input between images.

---

## Comparison: Semantic vs Instance Segmentation

This pipeline produces **semantic segmentation** (all road pixels share a single class label), as opposed to the SAM pipelines that produce **instance segmentation** (each road segment gets a unique ID).

| Aspect | Mask2Former (This Pipeline) | SAM 2 / SAM 3 |
|--------|-----------------------------|----------------|
| Output | Binary road/non-road mask | Instance mask with unique IDs |
| Individual segments | Not separated | Each segment has its own ID |
| Overlapping roads | Merged into one mask | Separated (watershed / SAM instances) |
| Polygon extraction | Not included | Included with metrics |
| Post-processing | Upscale only | Morphology + filtering + vectorization |
| Best for | Road coverage mapping, mask generation | Per-segment analysis, vectorization |

---

## Output Summary

| Output | Format | Resolution | Description |
|--------|--------|------------|-------------|
| Binary road mask | NumPy array (in-memory) | 2048×2048 | 1 = road, 0 = non-road |
| 3-panel visualization | matplotlib figure | 2048×2048 | Original + mask + overlay |

> **Note:** This pipeline does not export masks to disk or produce vector outputs. It is designed for visual inspection and mask generation. For polygon extraction and GeoJSON/Shapefile export, the SAM 2 (`User_Output.ipynb`) or SAM 3 (`Roads_Polygon_Extraction.ipynb`) pipelines should be used downstream.

---

## Libraries Used

| Library | Role |
|---------|------|
| `transformers` (Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation) | Model loading, preprocessing, inference, post-processing |
| `torch` | GPU-accelerated inference |
| `albumentations` | Image resizing (512×512) |
| `OpenCV` (cv2) | Nearest-neighbor upscaling (512 → 2048) |
| `numpy` | Mask array operations |
| `Pillow` (PIL) | Image loading and RGB conversion |
| `matplotlib` | 3-panel visualization |