import torch
import numpy as np
from PIL import Image

try:
    from torchvision.ops import nms
except ImportError:
    nms = None

def py_nms(boxes, scores, iou_threshold=0.45):
    """
    Pure NumPy implementation of Non-Maximum Suppression (fallback if torchvision is not present).
    boxes: Nx4 numpy array (x1, y1, x2, y2)
    scores: N numpy array
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def predict_tiled(
    image,
    yolo_model,
    tile_size=640,
    overlap=0.2,
    conf=0.25,
    iou_thresh=0.45
):
    """
    Divides full high-res image into overlapping tiles, runs YOLO on each tile,
    maps tile-relative coordinates back to full image coordinates, and removes
    duplicates using Non-Maximum Suppression (NMS).
    
    Args:
        image: PIL Image or numpy array (H, W, 3)
        yolo_model: Ultralytics YOLO model
        tile_size: int, width & height of sliding tile window (default 640)
        overlap: float, ratio of tile size for overlap (0.2 = 20% overlap)
        conf: float, YOLO confidence threshold
        iou_thresh: float, NMS IoU threshold for merging duplicates
        
    Returns:
        List of dicts: [{'box': [x1, y1, x2, y2], 'conf': float}]
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    h_img, w_img = img_np.shape[:2]

    stride = int(tile_size * (1.0 - overlap))
    if stride <= 0:
        stride = tile_size

    # Generate tile top-left coordinates (y_start, x_start)
    x_starts = list(range(0, w_img - tile_size + 1, stride))
    if len(x_starts) == 0 or x_starts[-1] + tile_size < w_img:
        x_starts.append(max(0, w_img - tile_size))

    y_starts = list(range(0, h_img - tile_size + 1, stride))
    if len(y_starts) == 0 or y_starts[-1] + tile_size < h_img:
        y_starts.append(max(0, h_img - tile_size))

    all_boxes = []
    all_scores = []

    for y_start in y_starts:
        for x_start in x_starts:
            x_end = min(w_img, x_start + tile_size)
            y_end = min(h_img, y_start + tile_size)

            tile = img_np[y_start:y_end, x_start:x_end]

            # Run YOLO on tile
            results = yolo_model.predict(tile, conf=conf, verbose=False)

            if results and len(results[0].boxes):
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for box, c_val in zip(xyxy, confs):
                    # Translate to full image coordinates
                    bx1, by1, bx2, by2 = box
                    fx1 = int(bx1 + x_start)
                    fy1 = int(by1 + y_start)
                    fx2 = int(bx2 + x_start)
                    fy2 = int(by2 + y_start)

                    # Clamp to image bounds
                    fx1 = max(0, min(w_img, fx1))
                    fy1 = max(0, min(h_img, fy1))
                    fx2 = max(0, min(w_img, fx2))
                    fy2 = max(0, min(h_img, fy2))

                    all_boxes.append([fx1, fy1, fx2, fy2])
                    all_scores.append(float(c_val))

    if len(all_boxes) == 0:
        return []

    boxes_arr = np.array(all_boxes, dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.float32)

    # Perform NMS
    if nms is not None:
        try:
            torch_boxes = torch.from_numpy(boxes_arr)
            torch_scores = torch.from_numpy(scores_arr)
            keep_indices = nms(torch_boxes, torch_scores, iou_thresh).numpy()
        except Exception:
            keep_indices = py_nms(boxes_arr, scores_arr, iou_thresh)
    else:
        keep_indices = py_nms(boxes_arr, scores_arr, iou_thresh)

    final_detections = []
    for idx in keep_indices:
        b = boxes_arr[idx].astype(int).tolist()
        c = float(scores_arr[idx])
        final_detections.append({'box': b, 'conf': c})

    # Sort detections left-to-right, top-to-bottom for clean output
    final_detections.sort(key=lambda d: (d['box'][1], d['box'][0]))
    return final_detections
