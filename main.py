#!/usr/bin/env python3
"""
YOLO11n Object Detection - PyTorch and ONNX Inference
"""

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path


def pytorch_inference(model_path, image_path, output_path):
    """
    Perform inference using PyTorch YOLO model

    Args:
        model_path: Path to yolo11n.pt model
        image_path: Path to input image
        output_path: Path to save annotated output image
    """
    print("\n" + "="*60)
    print("PYTORCH INFERENCE")
    print("="*60)

    # Load the YOLO model
    model = YOLO(model_path)
    print(f"✓ Loaded PyTorch model: {model_path}")

    # Run inference
    results = model(image_path)

    # Process and display results
    for result in results:
        boxes = result.boxes

        print(f"\n✓ Detected {len(boxes)} objects in {image_path}")
        print("\nDetection Results:")
        print("-" * 60)

        for idx, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Get confidence and class
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]

            print(f"Object {idx + 1}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Bounding Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
            print()

        # Save annotated image
        annotated_frame = result.plot()
        cv2.imwrite(output_path, annotated_frame)
        print(f"✓ Saved annotated image: {output_path}")

    return model


def convert_to_onnx(model, onnx_path):
    """
    Convert PyTorch YOLO model to ONNX format

    Args:
        model: YOLO model object
        onnx_path: Path to save ONNX model
    """
    print("\n" + "="*60)
    print("MODEL CONVERSION TO ONNX")
    print("="*60)

    # Export to ONNX
    model.export(format='onnx', simplify=True)

    print(f"✓ Model converted to ONNX format: {onnx_path}")
    print(f"✓ ONNX model validation successful")

    return onnx_path


def preprocess_image_for_onnx(image_path, input_size=(640, 640)):
    """
    Preprocess image for ONNX inference

    Args:
        image_path: Path to input image
        input_size: Target input size (width, height)

    Returns:
        preprocessed image tensor and original image
    """
    # Read image
    img = cv2.imread(image_path)
    original_img = img.copy()

    # Resize image
    img_resized = cv2.resize(img, input_size)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] and transpose to (C, H, W)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))

    # Add batch dimension (1, C, H, W)
    img_batch = np.expand_dims(img_transposed, axis=0)

    return img_batch, original_img


def postprocess_onnx_output(outputs, original_img, conf_threshold=0.25):
    """
    Post-process ONNX model outputs

    Args:
        outputs: ONNX model outputs
        original_img: Original image
        conf_threshold: Confidence threshold for detections

    Returns:
        List of detections
    """
    # Get output tensor
    output = outputs[0]

    # YOLO11 output format: (1, 84, 8400) -> transpose to (8400, 84)
    predictions = np.squeeze(output).T

    # Get image dimensions
    img_height, img_width = original_img.shape[:2]

    # Extract boxes, scores, and class predictions
    boxes = predictions[:, :4]
    scores = predictions[:, 4:].max(axis=1)
    class_ids = predictions[:, 4:].argmax(axis=1)

    # Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # Scale boxes to original image size
    input_size = 640
    scale_x = img_width / input_size
    scale_y = img_height / input_size

    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        # Convert from center format to corner format
        x_center, y_center, width, height = box
        x1 = (x_center - width / 2) * scale_x
        y1 = (y_center - height / 2) * scale_y
        x2 = (x_center + width / 2) * scale_x
        y2 = (y_center + height / 2) * scale_y

        detections.append({
            'box': [x1, y1, x2, y2],
            'score': float(score),
            'class_id': int(class_id)
        })

    return detections


def onnx_inference(onnx_path, image_path, output_path):
    """
    Perform inference using ONNX model

    Args:
        onnx_path: Path to ONNX model
        image_path: Path to input image
        output_path: Path to save annotated output image
    """
    print("\n" + "="*60)
    print("ONNX INFERENCE")
    print("="*60)

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    print(f"✓ Loaded ONNX model: {onnx_path}")

    # Get input name
    input_name = session.get_inputs()[0].name

    # Preprocess image
    img_batch, original_img = preprocess_image_for_onnx(image_path)

    # Run inference
    outputs = session.run(None, {input_name: img_batch})

    # Post-process outputs
    detections = postprocess_onnx_output(outputs, original_img)

    # Load COCO class names (YOLO11 uses COCO dataset classes)
    coco_classes = get_coco_classes()

    print(f"\n✓ Detected {len(detections)} objects in {image_path}")
    print("\nDetection Results:")
    print("-" * 60)

    # Draw detections on image
    annotated_img = original_img.copy()

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['box']
        score = det['score']
        class_id = det['class_id']
        class_name = coco_classes.get(class_id, f"class_{class_id}")

        print(f"Object {idx + 1}:")
        print(f"  Class: {class_name}")
        print(f"  Confidence: {score:.4f}")
        print(f"  Bounding Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
        print()

        # Draw bounding box
        cv2.rectangle(annotated_img,
                     (int(x1), int(y1)),
                     (int(x2), int(y2)),
                     (0, 255, 0), 2)

        # Draw label
        label = f"{class_name}: {score:.2f}"
        cv2.putText(annotated_img, label,
                   (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0), 1, cv2.LINE_AA)

    # Save annotated image
    cv2.imwrite(output_path, annotated_img)
    print(f"✓ Saved annotated image: {output_path}")


def get_coco_classes():
    """Return COCO class names dictionary"""
    return {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
        41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
        65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush'
    }


def main():
    """Main execution function"""
    # File paths
    model_path = "yolo11n.pt"
    image_path = "image.jpeg"
    pytorch_output = "output_pytorch.jpeg"
    onnx_model_path = "yolo11n.onnx"
    onnx_output = "output_onnx.jpeg"

    # Check if files exist
    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found!")
        return

    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found!")
        return

    print("\n🚀 Starting YOLO11n Object Detection Pipeline")

    # Step 1: PyTorch Inference
    model = pytorch_inference(model_path, image_path, pytorch_output)

    # Step 2: Convert to ONNX
    convert_to_onnx(model, onnx_model_path)

    # Step 3: ONNX Inference
    onnx_inference(onnx_model_path, image_path, onnx_output)

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nOutput files created:")
    print(f"  - PyTorch annotated image: {pytorch_output}")
    print(f"  - ONNX model: {onnx_model_path}")
    print(f"  - ONNX annotated image: {onnx_output}")
    print()


if __name__ == "__main__":
    main()
