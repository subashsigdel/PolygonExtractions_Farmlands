# Road Polygon Extraction with SAM 3 — Workflow

## Overview

The `Roads_Polygon_Extraction.ipynb` notebook implements a batch road segmentation pipeline using **SAM 3 (Segment Anything Model 3)** with text-prompted instance segmentation. The pipeline processes an entire folder of images, applies morphological post-processing, computes geometric metrics (including skeleton-based road length), and exports all results into a single combined **GeoJSON** file with globally unique IDs.

---

## Pipeline Architecture

```
Input Folder (roads/)
  ├── tile_10240_47104.tif
  ├── tile_10240_57344.tif
  └── ...
        │
        ▼
┌──────────────────────────────┐
│  STEP 1: Load SAM 3 Model    │  ← facebook/sam3 (bfloat16, CUDA)
└───────────┬──────────────────┘
            ▼
┌──────────────────────────────┐
│  STEP 2: Batch Segmentation  │  ← Text prompt: "road"
│  Per-image processing:       │
│    ├─ Preprocess (Sam3Proc)  │
│    ├─ Inference (no_grad)    │
│    ├─ Post-process instances │
│    ├─ Instance mask creation │
│    ├─ Morphological cleanup  │
│    │   ├─ Closing (×3)       │
│    │   ├─ Opening            │
│    │   ├─ Gaussian blur ×2   │
│    └─ Save GeoTIFF mask      │
└───────────┬──────────────────┘
            ▼
┌──────────────────────────────┐
│  STEP 3: Polygon Extraction  │
│    ├─ rasterio.features      │
│    ├─ Per-segment metrics    │
│    │   ├─ Bounding rectangle │
│    │   ├─ Skeleton length    │
│    │   ├─ Aspect ratio       │
│    │   └─ Elongation         │
│    └─ Global ID assignment   │
└───────────┬──────────────────┘
            ▼
┌──────────────────────────────┐
│  STEP 4: Combined GeoJSON    │  ← all_roads_combined.geojson
└───────────┬──────────────────┘
            ▼
┌──────────────────────────────┐
│  STEP 5: Visualization       │
│    ├─ Single pair view       │
│    ├─ Multiple pair view     │
│    └─ Grid view              │
└──────────────────────────────┘
```

---

## Step-by-Step Workflow

### Step 1 — SAM 3 Model Loading

SAM 3 is loaded from HuggingFace in bfloat16 precision for memory-efficient GPU inference:

```python
from transformers import Sam3Model, Sam3Processor

def load_sam3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained(
        "facebook/sam3",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    return model, processor, device
```

---

### Step 2 — Per-Image Segmentation with Text Prompt

Each image is processed individually through `process_image_sam3()`:

**2a. Image Loading & Preprocessing**

```python
image = Image.open(image_path).convert("RGB")

inputs = processor(
    images=image,
    text="road",          # Text prompt — targets road structures specifically
    return_tensors="pt"
).to(device)
```

The text prompt `"road"` directs SAM 3 to only segment objects matching this description, significantly reducing false positives compared to automatic mode.

**2b. Inference**

```python
with torch.no_grad():
    outputs = model(**inputs)
```

**2c. Instance Post-Processing**

```python
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.55–0.60,      # Instance confidence threshold
    mask_threshold=0.40–0.42, # Per-pixel binary threshold
    target_sizes=inputs["original_sizes"].tolist()
)[0]
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `threshold` (score) | 0.55–0.60 | Minimum confidence to keep an instance; higher = fewer but more confident detections |
| `mask_threshold` | 0.40–0.42 | Threshold for converting soft mask probabilities to binary; lower = more inclusive masks |

**2d. Instance Mask Creation**

```python
def sam3_to_instance_mask(masks):
    masks = masks.float().cpu().numpy()
    instance_mask = np.zeros((h, w), dtype=np.uint16)
    
    for i, m in enumerate(masks, start=1):
        instance_mask[m > 0.5] = i   # Each mask gets unique ID (1, 2, 3, ...)
    
    return instance_mask
```

Later masks overwrite earlier ones in overlapping regions (last-write-wins).

**2e. Morphological Post-Processing**

Each instance undergoes a multi-step cleanup:

```
For each instance:
    ├─ Morphological Closing × 3 iterations (kernel: 7×7 ellipse)
    │     → Fills gaps, connects broken segments
    ├─ Morphological Opening × 1 (kernel: 3×3 ellipse)
    │     → Removes small noise protrusions
    ├─ Gaussian Blur #1 (7×7 kernel, σ=1.5) + threshold at 0.5
    │     → First smoothing pass for edge refinement
    └─ Gaussian Blur #2 (5×5 kernel, σ=0.8) + threshold at 0.5
          → Second lighter smoothing for final polish
```

| Operation | Kernel Size | Iterations/σ | Purpose |
|-----------|------------|--------------|---------|
| Closing | 7×7 ellipse | 3 iterations | Fill internal holes and bridge small gaps |
| Opening | 3×3 ellipse | 1 iteration | Remove noise pixels and thin protrusions |
| Gaussian Blur #1 | 7×7 | σ = 1.5 | Major edge smoothing |
| Gaussian Blur #2 | 5×5 | σ = 0.8 | Fine edge refinement |

**Why two Gaussian passes?** A single aggressive blur can over-smooth thin road segments. The two-pass approach (coarse then fine) achieves smooth boundaries while preserving narrow road geometry.

**2f. GeoTIFF Export**

```python
with rasterio.open(mask_path, "w", driver="GTiff",
                   height=h, width=w, count=1,
                   dtype=instance_mask.dtype,
                   transform=transform, crs=crs,
                   compress='lzw') as dst:
    dst.write(instance_mask, 1)
```

Georeferencing (CRS + affine transform) is read from the source image and written to the output mask, ensuring spatial alignment.

---

### Step 3 — Polygon Extraction with Geometric Metrics

**3a. Vectorization**

Each instance mask is converted to vector polygons using `rasterio.features.shapes`:

```python
def extract_polygons_per_object(instance_mask, transform, global_id_offset, image_name):
    for inst_id in unique_ids:
        obj_mask = (instance_mask == inst_id).astype(np.uint8)
        
        for geom, val in shapes(obj_mask, mask=obj_mask, transform=transform):
            if val > 0:
                global_id = global_id_offset + int(inst_id)
                # Store polygon with global ID, metrics, and source info
                break
```

**Global ID system:** A running `global_id_offset` ensures that IDs are unique across all images in the batch. Image 1 might have IDs 1–15, image 2 gets 16–28, etc.

**3b. Road Segment Metrics**

For each road segment, `calculate_road_metrics()` computes:

| Metric | Computation | What It Measures |
|--------|------------|-----------------|
| **Length** | `max(w, h)` from `cv2.minAreaRect` | Longest dimension of minimum bounding rectangle |
| **Width** | `min(w, h)` from `cv2.minAreaRect` | Shortest dimension of minimum bounding rectangle |
| **Aspect Ratio** | `length / width` | How elongated the road segment is |
| **Skeleton Length** | `np.sum(skeletonize(obj_mask))` | Pixel count of the medial axis (road centerline proxy) |
| **Area** | `np.count_nonzero(obj_mask)` | Total number of road pixels |
| **Elongation** | `skeleton_length / √area` | Normalized measure of how stretched the segment is |

**Skeletonization** (`skimage.morphology.skeletonize`) reduces each road segment to its 1-pixel-wide medial axis. The total skeleton pixel count serves as a robust estimate of road centerline length, which is more meaningful than bounding rectangle length for curved roads.

---

### Step 4 — Combined GeoJSON Export

All polygons from all images are merged into a single GeoJSON FeatureCollection:

```python
def save_combined_geojson(all_polygons, output_path, crs):
    features = []
    for p in all_polygons:
        properties = {
            "id": p['global_id'],
            "local_id": p['local_id'],
            "area_pixels": p['area_pixels'],
            "source_image": p['source_image'],
            "length_pixels": p['length_pixels'],
            "width_pixels": p['width_pixels'],
            "aspect_ratio": p['aspect_ratio'],
            "elongation": p['elongation'],
            "skeleton_length": p['skeleton_length'],
            "length_width_ratio": p['length_width_ratio']
        }
        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": mapping(p['polygon'])
        })
```

**Output file:** `all_roads_combined.geojson`

**Per-feature properties:**

| Property | Type | Description |
|----------|------|-------------|
| `id` | int | Globally unique ID across all images |
| `local_id` | int | ID within the source image |
| `source_image` | string | Name of the source tile |
| `area_pixels` | int | Road segment area in pixels |
| `length_pixels` | float | Length from min bounding rectangle |
| `width_pixels` | float | Width from min bounding rectangle |
| `aspect_ratio` | float | length / width |
| `elongation` | float | skeleton_length / √area |
| `skeleton_length` | float | Medial axis pixel count |
| `length_width_ratio` | float | Same as aspect_ratio |

---

### Step 5 — Visualization

Three visualization modes are provided:

**Single Pair View** (`visualize_single_pair`): Displays one image alongside its colored instance mask. Automatically resizes large images (>1024px) for display while preserving instance IDs via nearest-neighbor interpolation.

**Multiple Pair View** (`visualize_multiple_pairs`): Iterates through all image-mask pairs in a folder, displaying each as a side-by-side plot with original size annotation and segment count. Optionally saves each visualization as a PNG.

**Grid View** (`visualize_grid`): Shows up to 9 image-mask pairs in a compact grid layout for batch review.

Instance masks are rendered using the `nipy_spectral` colormap, giving each segment a distinct color for easy visual differentiation.

---

## Complete Pipeline Execution

```python
run_complete_pipeline(
    input_folder="roads",                          # Folder with GeoTIFF tiles
    output_mask_folder="outputs_sam3",             # Where instance masks are saved
    output_geojson="all_roads_combined.geojson",   # Combined vector output
    text_prompt="road",                            # SAM 3 text prompt
    score_threshold=0.60,                          # Instance confidence
    mask_threshold=0.42                            # Pixel threshold
)
```

**Pipeline summary output:**
```
Input images:           N
Successfully segmented: M
Mask files created:     M
Total polygons:         P
Total road segments:    S
Average per image:      S/M
Global ID range:        1 to max_id
```

---

## Output Files Summary

| File | Location | Format | Contents |
|------|----------|--------|----------|
| Instance masks | `outputs_sam3/*.tif` | GeoTIFF (uint16, LZW) | Per-image instance masks with CRS |
| Combined polygons | `all_roads_combined.geojson` | GeoJSON | All road polygons with metrics |
| Visualizations | `visualizations/*.png` | PNG | Side-by-side image/mask plots |

---

## Libraries Used

| Library | Role |
|---------|------|
| `transformers` (Sam3Model, Sam3Processor) | SAM 3 model loading, text-prompted inference |
| `torch` | GPU inference with bfloat16 precision |
| `rasterio` | GeoTIFF I/O, CRS preservation, feature vectorization |
| `OpenCV` (cv2) | Morphological ops, Gaussian blur, contours, min bounding rect |
| `shapely` | Polygon geometry and GeoJSON serialization |
| `skimage.morphology` | Skeletonization for centerline length |
| `matplotlib` | Visualization with colormapped instance masks |
| `numpy` | Mask array operations |
| `Pillow` (PIL) | Image loading |