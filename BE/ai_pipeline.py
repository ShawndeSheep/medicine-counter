# Imports
from dotenv import load_dotenv
import os
import torch
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam3Processor, Sam3Model, AutoImageProcessor, AutoModel
import hdbscan
import umap

# Init
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

# SAM3 Model Init
_SAM3_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_SAM3_MODEL = None
_SAM3_PROCESSOR = None
try:
    if HF_TOKEN:
        _SAM3_MODEL = Sam3Model.from_pretrained("facebook/sam3", token=HF_TOKEN).to(_SAM3_DEVICE)
        _SAM3_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3", token=HF_TOKEN)
    else:
        _SAM3_MODEL = Sam3Model.from_pretrained("facebook/sam3").to(_SAM3_DEVICE)
        _SAM3_PROCESSOR = Sam3Processor.from_pretrained("facebook/sam3")
except Exception as _e:
    print("SAM3 unavailable", _e)

# DINOv3 Model Init
_DINOV3_MODEL = None
_DINOV3_PROCESSOR = None
try:
    if HF_TOKEN:
        _DINOV3_PROCESSOR = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", token=HF_TOKEN)
        _DINOV3_MODEL = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", token=HF_TOKEN).to(_SAM3_DEVICE)
    else:
        _DINOV3_PROCESSOR = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        _DINOV3_MODEL = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(_SAM3_DEVICE)
except Exception as _e:
    print("DINOv3 unavailable", _e)


def start_pipeline(image):
    print("Running AI pipeline...")
    total_start = time.time()
    
    # 1. Segmentation
    seg_start = time.time()
    segmentation_result = segmentation(image)
    seg_time = time.time() - seg_start
    print(f"[Segmentation] Found {segmentation_result['count']} objects | Time: {seg_time:.2f}s")
    show_segmentation(segmentation_result)

    # 2. Embedding
    emb_start = time.time()
    embedding_result = embedding(segmentation_result)
    emb_time = time.time() - emb_start
    print(f"[Embedding] Generated {embedding_result['count']} embeddings | Time: {emb_time:.2f}s")
    
    # 3. Clustering
    clust_start = time.time()
    clustering_result = clustering(embedding_result)
    clust_time = time.time() - clust_start
    print(f"[Clustering] {clustering_result} | Time: {clust_time:.2f}s")
    
    # 4. Classification
    class_start = time.time()
    classification_result = classification()
    class_time = time.time() - class_start
    print(f"[Classification] {classification_result} | Time: {class_time:.2f}s")
    
    # 5. Output Formatting
    outputFormatting_result = outputFormatting()
    
    total_time = time.time() - total_start
    print(f"\n[Pipeline Complete] Total time: {total_time:.2f}s")
    print(f"  - Segmentation: {seg_time:.2f}s")
    print(f"  - Embedding: {emb_time:.2f}s")
    print(f"  - Clustering: {clust_time:.2f}s")
    print(f"  - Classification: {class_time:.2f}s")
    
    return outputFormatting_result


def show_segmentation(result):
    """
    Display the segmentation result in a matplotlib popup window.
    
    Args:
        result: Dictionary from segmentation() containing 'visualization' and 'count'
    """
    visualization = result.get("visualization")
    count = result.get("count", 0)
    
    if visualization is None:
        print("No visualization available")
        return
    
    plt.figure(figsize=(10, 8))
    plt.imshow(visualization)
    plt.title(f"Segmentation Result: {count} objects detected")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def overlay_masks(image, masks):
    """
    Overlay colored masks on an image for visualization.
    
    Args:
        image: PIL Image to overlay masks on
        masks: Tensor of binary masks [N, H, W]
    
    Returns:
        PIL Image with colored mask overlays
    """
    image = image.convert("RGBA")
    masks_np = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks_np.shape[0]
    if n_masks == 0:
        return image
    
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks_np, colors):
        mask_img = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    
    return image


def segmentation(image, threshold=0.5, mask_threshold=0.5, show_mask=True):
    """
    Perform instance segmentation on an image using SAM3.
    
    Args:
        image: PIL Image or path to image file
        threshold: Confidence threshold for detections (default: 0.5)
        mask_threshold: Threshold for binary mask creation (default: 0.5)
        show_mask: Whether to generate mask visualization (default: True)
    
    Returns:
        dict containing:
            - masks: Binary masks for each detected object
            - boxes: Bounding boxes in xyxy format
            - scores: Confidence scores for each detection
            - count: Number of objects detected
            - visualization: PIL Image with mask overlay (if show_mask=True)
    """
    if _SAM3_MODEL is None or _SAM3_PROCESSOR is None:
        raise Exception("SAM3 model or processor not initialized")
    
    # Load image based on input type
    if isinstance(image, str):
        # File path
        image = Image.open(image).convert("RGB")
    elif hasattr(image, 'read'):
        # Flask FileStorage or file-like object
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        # Assume it's a numpy array
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    
    # Text prompt for medicine-related objects
    prompt = "Pills, Capsules, Tablets, Medicines"
    
    # Process inputs
    inputs = _SAM3_PROCESSOR(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(_SAM3_DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = _SAM3_MODEL(**inputs)
    
    # Post-process results
    results = _SAM3_PROCESSOR.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    masks = results.get("masks", torch.tensor([]))
    boxes = results.get("boxes", torch.tensor([]))
    scores = results.get("scores", torch.tensor([]))
    
    count = len(masks) if masks is not None else 0
    print(f"Found {count} objects")
    
    # Build result dictionary
    result = {
        "masks": masks,
        "boxes": boxes,
        "scores": scores,
        "count": count,
        "original_image": image  # Pass original image for embedding step
    }
    
    # Generate visualization if requested
    if show_mask and count > 0:
        visualization = overlay_masks(image, masks)
        result["visualization"] = visualization
    else:
        result["visualization"] = image.convert("RGBA") if show_mask else None
    
    return result

def embedding(segmentation_result):
    """
    Generate DINOv3 CLS token embeddings for each segmented object.
    
    Args:
        segmentation_result: Dictionary from segmentation() containing:
            - masks: Binary masks for each detected object
            - boxes: Bounding boxes in xyxy format
            - original_image: The original PIL Image
    
    Returns:
        dict containing:
            - embeddings: Tensor of shape [N, hidden_size] with CLS embeddings for each object
            - count: Number of embeddings generated
    """
    if _DINOV3_MODEL is None or _DINOV3_PROCESSOR is None:
        raise Exception("DINOv3 model or processor not initialized")
    
    boxes = segmentation_result.get("boxes")
    original_image = segmentation_result.get("original_image")
    
    if original_image is None:
        raise Exception("Original image not found in segmentation result")
    
    if boxes is None or len(boxes) == 0:
        print("No objects to embed")
        return {
            "embeddings": torch.tensor([]),
            "count": 0
        }
    
    embeddings_list = []
    
    # Process each segmented object
    for box in boxes:
        # Extract bounding box coordinates (xyxy format)
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Crop the object from the original image
        cropped_image = original_image.crop((x1, y1, x2, y2))
        
        # Process the cropped image
        inputs = _DINOV3_PROCESSOR(images=cropped_image, return_tensors="pt").to(_SAM3_DEVICE)
        
        # Get embeddings
        with torch.inference_mode():
            outputs = _DINOV3_MODEL(**inputs)
        
        # Extract CLS token (first token)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Shape: [1, hidden_size]
        embeddings_list.append(cls_token)
    
    # Stack all embeddings
    embeddings = torch.cat(embeddings_list, dim=0)  # Shape: [N, hidden_size]
    
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    return {
        "embeddings": embeddings,
        "count": len(embeddings),
        "boxes": boxes
    }

def clustering(embedding_result):
    """
    Perform HDBSCAN clustering on embeddings.
    
    Args:
        embedding_result: Dictionary from embedding() containing:
            - embeddings: Tensor of shape [N, hidden_size]
            - boxes: Bounding boxes for each object
    
    Returns:
        dict containing:
            - labels: Cluster labels for each embedding (-1 = noise)
            - n_clusters: Number of clusters found
            - embeddings: Original embeddings
            - embeddings_2d: UMAP-reduced 2D embeddings for visualization
            - boxes: Original bounding boxes
    """
    embeddings = embedding_result.get("embeddings")
    boxes = embedding_result.get("boxes")
    
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings to cluster")
        return {
            "labels": np.array([]),
            "n_clusters": 0,
            "embeddings": embeddings,
            "embeddings_2d": np.array([]),
            "boxes": boxes
        }
    
    # Convert to numpy if tensor
    if torch.is_tensor(embeddings):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Reduce dimensionality with UMAP for visualization and potentially better clustering
    n_samples = len(embeddings_np)
    n_neighbors = min(15, n_samples - 1) if n_samples > 1 else 1
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42
    )
    embeddings_2d = reducer.fit_transform(embeddings_np)
    
    # Perform HDBSCAN clustering
    min_cluster_size = max(2, n_samples // 10)  # At least 2, or 10% of samples
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_epsilon=0.0
    )
    labels = clusterer.fit_predict(embeddings_2d)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")
    
    result = {
        "labels": labels,
        "n_clusters": n_clusters,
        "embeddings": embeddings,
        "embeddings_2d": embeddings_2d,
        "boxes": boxes
    }
    
    # Show clustering visualization
    show_clustering(result)
    
    return result


def show_clustering(result):
    """
    Display the clustering result in a matplotlib scatter plot.
    
    Args:
        result: Dictionary from clustering() containing 'embeddings_2d' and 'labels'
    """
    embeddings_2d = result.get("embeddings_2d")
    labels = result.get("labels")
    n_clusters = result.get("n_clusters", 0)
    
    if embeddings_2d is None or len(embeddings_2d) == 0:
        print("No data to visualize")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Create color palette
    unique_labels = set(labels)
    cmap = matplotlib.colormaps.get_cmap("tab10")
    
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Noise points in gray
            color = 'lightgray'
            marker = 'x'
            label_name = 'Noise'
        else:
            color = cmap(label % 10)
            marker = 'o'
            label_name = f'Cluster {label}'
        
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            marker=marker,
            s=60,
            alpha=0.7,
            label=label_name,
            edgecolors='white',
            linewidths=0.5
        )
    
    plt.title(f"HDBSCAN Clustering Result: {n_clusters} clusters found")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def classification():
    return "Running classification..."

def outputFormatting():
    return "Running output formatting..."