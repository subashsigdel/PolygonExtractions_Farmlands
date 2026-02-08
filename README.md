# FarmLand Mask to Polygon Shapefile Pipeline — Complete Workflow

## Overview

The `FarmMasks_pipeline.ipynb` notebook implements a fully interactive, end-to-end pipeline that takes a raw farmLands (size = 2048*2048) image as input, generates segmentation masks using **SAM 2 (Segment Anything Model 2)**, applies extensive post-processing to clean and refine the masks, extracts vector polygons, and provides an interactive visualization where the user can select individual segments and export them as **Shapefiles**.

---

## Pipeline Architecture

```
User Input (image path)
        │
        ▼
┌─────────────────────────┐
│  SAM 2 Mask Generation  │  ← sam2-hiera-large (samgeo)
│  (Automatic Mode)       │
└───────────┬─────────────┘
            │  masks.tif
            ▼
┌─────────────────────────┐
│  Mask Loading &         │  ← rasterio + scipy.ndimage
│  Instance Conversion    │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Watershed Separation   │  ← cv2.distanceTransform +  cv2.  watershed 
|         
│  (Split Touching Objs)  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Morphological Ops      │  ← Closing → Dilation → Erosion
│  (Shape Improvement)    │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Shape Quality Filtering│  ← 6 geometric metrics
│ (Remove False Positives)│
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Polygon Extraction     │  ← rasterio.features.shapes + Shapely
│  + GeoJSON Export       │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Boundary Extraction    │  ← Morphological gradient
│  + GeoTIFF Export       │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Interactive Visualize  │  ← matplotlib + geopandas
│  + Shapefile Export     │
└─────────────────────────┘
```

---

## Step-by-Step Workflow

### Step 1 — SAM 2 Model Initialization

The pipeline begins by initializing the **SAM 2** model using the `samgeo` library's `SamGeo2` wrapper, which adds geospatial awareness (CRS preservation, GeoTIFF I/O) on top of Meta's SAM 2.

```python
from samgeo import SamGeo2

sam2 = SamGeo2(
    model_id="sam2-hiera-large",
    apply_postprocessing=False,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.85,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.9,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)
```

**Key parameters explained:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `model_id` | `sam2-hiera-large` | Uses the largest Hiera vision encoder for best quality |
| `apply_postprocessing` | `False` | Disables SAM's internal post-processing (custom pipeline handles it) |
| `points_per_side` | `32` | Places a 32×32 grid of automatic point prompts (1,024 total) |
| `points_per_batch` | `64` | Processes 64 points per forward pass for memory efficiency |
| `pred_iou_thresh` | `0.6` | Minimum predicted IoU to keep a mask (filters low-confidence) |
| `stability_score_thresh` | `0.85` | Minimum stability score — ensures masks are consistent across thresholds |
| `stability_score_offset` | `0.7` | Offset used in stability score computation |
| `crop_n_layers` | `1` | Number of crop layers for multi-scale detection |
| `box_nms_thresh` | `0.9` | Non-maximum suppression threshold for overlapping boxes |
| `crop_n_points_downscale_factor` | `2` | Reduces point density in crop layers |
| `min_mask_region_area` | `25.0` | Minimum mask region in pixels (removes tiny detections) |
| `use_m2m` | `True` | Enables mask-to-mask refinement for higher quality |

---

### Step 2 — Interactive Pipeline Execution

The `pipeline()` function is the entry point. It prompts the user for an image path, generates masks, and runs the full processing chain:

```python
def pipeline():
    image_path = input("Enter the path to your image: ").strip()
    
    # Generate mask using SAM 2
    sam2.generate(image_path)
    sam2.save_masks(output="masks.tif")
    
    # Run post-processing + polygon extraction
    polygons_geojson = main(mask_path="masks.tif")
    
    # Visualize and export
    Visualize(geojson_path=polygons_geojson, original_image_path=image_path)
```

**What happens:**
1. User provides path to a GeoTIFF satellite tile (e.g., `tiff_testing/tile_20480_40960.tif`)
2. SAM 2 generates automatic masks and saves to `masks.tif`
3. `main()` runs all post-processing and exports GeoJSON
4. `Visualize()` displays results and allows interactive polygon selection

---

### Step 3 — Mask Loading & Instance Conversion

The `load_sam_mask()` function handles SAM's output format, which can be either multi-band or single-band:

```python
def load_sam_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask_data = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
```

**Multi-band handling:** If SAM outputs multiple bands (one per detected object), each band's non-zero pixels are assigned a unique instance ID in a single uint16 array. First-come-first-served — earlier bands take priority in overlapping regions.

**Single-band handling:** If it's a single binary mask, `scipy.ndimage.label` performs connected component labeling to assign unique IDs to each spatially disconnected region.

---

### Step 4 — Watershed Separation

**Problem:** SAM sometimes merges touching objects into a single mask region.

**Solution:** The watershed algorithm separates them:

```python
def apply_watershed_separation(instance_mask):
    binary_mask = (instance_mask > 0).astype(np.uint8)
    
    # 1. Distance transform — pixels farther from edges get higher values
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
    
    # 2. Find peaks (object centers) at 20% of max distance
    _, peaks = cv2.threshold(dist_norm, 0.2 * dist_norm.max(), 255, cv2.THRESH_BINARY)
    
    # 3. Label peaks as watershed markers
    _, markers = cv2.connectedComponents(peaks)
    
    # 4. Apply watershed to split merged regions
    markers = cv2.watershed(binary_3ch, markers)
    
    # 5. Boundaries (marker == -1) become background
    separated_mask = np.where(markers > 0, markers, 0).astype(np.uint16)
```

**Configuration:** `WATERSHED_THRESHOLD = 0.2` — lower values create more aggressive separation. Range: 0.1 (very aggressive) to 0.5 (conservative).

---

### Step 5 — Morphological Operations

Each instance is processed individually with three sequential operations:

```
Original → Closing → Dilation → Erosion → Cleaned
```

| Operation | Kernel Shape | Kernel Size | Iterations | Purpose |
|-----------|-------------|-------------|------------|---------|
| **Closing** | Elliptical | 2×2 | 1 | Fill small holes and gaps inside objects |
| **Dilation** | Elliptical | 3×3 | 1 | Expand objects, smooth jagged edges |
| **Erosion** | Elliptical | 3×3 | 1 | Shrink back to approximate original size with cleaner boundary |

**Why per-instance processing?** Applying morphology globally would cause nearby instances to merge. Processing each instance separately in its own binary mask prevents this.

---

### Step 6 — Shape Quality Filtering

The `filter_objects_by_quality()` function computes six geometric metrics from OpenCV contour analysis and removes objects that fail any threshold:

| Metric | Formula | Threshold | What It Removes |
|--------|---------|-----------|-----------------|
| **Area** | `cv2.contourArea(contour)` | 2,000 – 5,000,000 px | Tiny noise and massive background blobs |
| **Compactness** | `4π·area / perimeter²` | > 0.01 | Highly irregular, jagged shapes |
| **Solidity** | `area / convex_hull_area` | > 0.2 | Fragmented shapes with deep concavities |
| **Aspect Ratio** | `max(w,h) / min(w,h)` | < 200.0 | Extremely thin linear artifacts |
| **Convexity** | `hull_perimeter / perimeter` | > 0.3 | Very concave, complex boundary shapes |
| **Extent** | `area / bbox_area` | > 0.05 | Sparse shapes that barely fill their bounding box |

**Filtering is sequential:** An object is removed at the first failed metric (checked in the order listed above).

---

### Step 7 — Polygon Extraction & GeoJSON Export

Each surviving instance is vectorized to a polygon:

```python
def extract_polygons_per_object(instance_mask, transform):
    for inst_id in unique_ids:
        obj_mask = (instance_mask == inst_id).astype(np.uint8)
        
        for geom, val in shapes(obj_mask, mask=obj_mask, transform=transform):
            if val > 0:
                poly = shape(geom)
                polygons.append({'polygon': poly, 'id': int(inst_id), 'area': int(area)})
                break  # One polygon per instance (largest)
```

**Key details:**
- `rasterio.features.shapes` converts raster regions → Shapely polygon geometries
- The `transform` parameter ensures polygons are in the image's coordinate space (UTM)
- Only the first (largest) polygon per instance is kept
- Output saved as `polygons.geojson` with properties: `id`, `area_pixels`

---

### Step 8 — Boundary Extraction

Object boundaries are extracted for visualization using a morphological gradient:

```python
def extract_boundaries(instance_mask, thickness=4):
    for inst_id in unique_ids:
        obj_mask = (instance_mask == inst_id).astype(np.uint8)
        edges = cv2.morphologyEx(obj_mask, cv2.MORPH_GRADIENT, kernel)  # dilation - erosion
        boundaries = np.maximum(boundaries, edges)
    
    # Thicken boundaries for visibility
    boundaries = cv2.dilate(boundaries, thick_kernel, iterations=1)
```

**Outputs:**
- `masks_clean1.tif` — Cleaned uint16 instance mask (LZW compressed)
- `masks_boundaries1.tif` — uint8 boundary edge map

---

### Step 9 — Interactive Visualization & Shapefile Export

The `Visualize()` function provides an interactive workflow:

**Display phase:**
1. Load the original GeoTIFF image and compute display extent from its affine transform
2. Load the GeoJSON polygons into a GeoDataFrame (geopandas)
3. Overlay polygon boundaries in **red** (linewidth=2) on the satellite image
4. Annotate each polygon with its **ID** at the centroid (blue, bold, fontsize=10)
5. Display using matplotlib

**Export phase:**
1. User enters a polygon ID via `input()` prompt
2. The selected polygon's GeoJSON is extracted from the GeoDataFrame
3. Column names are truncated to 10 characters (shapefile format limitation)
4. CRS is set to **EPSG:32645** (UTM Zone 45N)
5. Exported as `output.shp` using geopandas `to_file()` with ESRI Shapefile driver

---

## Output Files Summary

| File | Format | Contents |
|------|--------|----------|
| `masks.tif` | GeoTIFF | Raw SAM 2 output mask |
| `masks_clean1.tif` | GeoTIFF (uint16, LZW) | Post-processed instance mask with unique IDs |
| `masks_boundaries1.tif` | GeoTIFF (uint8) | Object boundary edge map |
| `polygons.geojson` | GeoJSON | All polygons with id and area_pixels |
| `output.shp` | ESRI Shapefile | Single user-selected polygon (CRS: EPSG:32645) |

---

## Libraries Used

| Library | Version Context | Role |
|---------|----------------|------|
| `samgeo` (SamGeo2) | SAM 2 wrapper | Geospatial-aware automatic mask generation |
| `rasterio` | GeoTIFF I/O | Read/write masks with CRS and transform metadata |
| `OpenCV` (cv2) | Image processing | Morphology, watershed, contours, distance transform |
| `scipy.ndimage` | Labeling | Connected component labeling for binary→instance |
| `shapely` | Geometry | Polygon creation and GeoJSON serialization |
| `geopandas` | Vector data | GeoDataFrame ops, shapefile export, CRS assignment |
| `matplotlib` | Visualization | Image display with polygon overlay |
| `numpy` | Array ops | Mask manipulation throughout the pipeline |
| `Pillow` (PIL) | Image loading | RGB conversion for display |