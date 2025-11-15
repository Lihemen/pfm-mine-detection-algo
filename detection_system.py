"""
PFM-1 Mine Object Detection and Tracking System
Using YOLOv8 for detection and ByteTrack for tracking
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from collections import defaultdict
import json

from ultralytics import YOLO

class PFM1DetectionSystem:
  """Complete system for PFM-1 mine detection and tracking"""
  def __init__(self, data_path='./data', model_size='yolov8n'):
    """
        Initialize the detection system
        
        Args:
          data_path: Path to dataset directory
          model_size: YOLOv8 model size (n/s/m/l/x)
    """
    self.data_path = Path(data_path)
    self.model_size = model_size
    self.model = None
    self.results_history = []
  
  def setup_dataset_structure(self):
    """Create the required dataset directory structure"""
    print("📁 Setting up dataset structure...")
    
    # Create directories
    dirs = [
      self.data_path / 'images' / 'train',
      self.data_path / 'images' / 'validation',
      self.data_path / 'images' / 'test',
      self.data_path / 'labels' / 'train',
      self.data_path / 'labels' / 'validation',
      self.data_path / 'labels' / 'test',
    ]
    
    for d in dirs:
      d.mkdir(parents=True, exist_ok=True)

    print(f"✅ Created dataset structure at {self.data_path}")
    print("\n📝 Next steps:")
    print("1. Place your images in the images/train, images/validation, and images/test folders")
    print("2. Create YOLO format labels (class x_center y_center width height) normalized to 0-1")
    print("3. Place corresponding .txt label files in labels/train, labels/validation, labels/test")
    
    return dirs
  
  def create_data_yaml(self):
    """Create YOLO dataset configuration file"""
    print("\n📄 Creating data.yaml configuration...")
    
    data_config = {
        'path': str(self.data_path.absolute()),
        'train': 'images/train',
        'val': 'images/validation',
        'test': 'images/test',
        'nc': 1,  # number of classes
        'names': ['PFM-1']  # class names
    }
    
    yaml_path = self.data_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
      yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created {yaml_path}")
    return yaml_path
  
  def train_model(self, epochs=100, imgsz=640, batch=16, patience=20):
    """
    Train YOLOv8 model on PFM-1 dataset
    
    Args:
      epochs: Number of training epochs
      imgsz: Image size for training
      batch: Batch size
      patience: Early stopping patience
    """
    print(f"\n🚀 Starting training with {self.model_size}...")
    
    # Initialize model
    self.model = YOLO(f'{self.model_size}.pt')
    
    # Train
    results = self.model.train(
        data=str(self.data_path / 'data.yaml'),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        save=True,
        project='training',
        name='exp',
        plots=True,
        device='cpu',  # Use GPU if available, else CPU
        # Augmentation parameters for better generalization
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=10,   # Rotation augmentation
        translate=0.1, # Translation augmentation
        scale=0.5,    # Scale augmentation
        flipud=0.0,   # No vertical flip (mines have orientation)
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
    )
    
    print("✅ Training complete!")
    return results
  
  def load_model(self, model_path = None):
    """Load a trained model"""
    print(f"\n📥 Loading model from {model_path}...")
    self.model_path = model_path if model_path is not None else f"{self.model_size}.pt"
    self.model = YOLO(self.model_path)
    print("✅ Model loaded successfully!")
    
  def validate_model(self):
    """Validate model on validation set and calculate metrics"""
    print("\n📊 Validating model...")
    
    if self.model is None:
      raise ValueError("No model loaded! Train or load a model first.")
    
    metrics = self.model.val(
      data=str(self.data_path / 'data.yaml'),
      plots=True,
      save_json=True,
    )
    
    # Extract key metrics
    results = {
      'mAP50': float(metrics.box.map50),
      'mAP50-95': float(metrics.box.map),
      'precision': float(metrics.box.mp),
      'recall': float(metrics.box.mr),
      'f1_score': 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr) if (metrics.box.mp + metrics.box.mr) > 0 else 0
    }
    
    print("\n📈 Validation Results:")
    print(f"  mAP@0.5: {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {results['mAP50-95']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    
    return results
  
  def test_on_images(self, image_folder, conf_threshold=0.25, save_results=True):
    """
    Run detection on test images
    
    Args:
      image_folder: Path to folder containing test images
      conf_threshold: Confidence threshold for detections
      save_results: Whether to save annotated images
    """
    print(f"\n🔍 Testing on images in {image_folder}...")
    
    if self.model is None:
      raise ValueError("No model loaded!")
    
    image_folder = Path(image_folder)
    results_list = []
    
    # Get all image files
    image_files = list(image_folder.glob('*.jpg')) + \
                  list(image_folder.glob('*.jpeg')) + \
                  list(image_folder.glob('*.png'))
    
    output_dir = Path('detection_results')
    output_dir.mkdir(exist_ok=True)
    
    for img_path in image_files:
      results = self.model.predict(
        source=str(img_path),
        conf=conf_threshold,
        save=save_results,
        project=str(output_dir),
        name='predictions',
        exist_ok=True,
      )
        
      # Store results
      detections = []
      for r in results:
        boxes = r.boxes
        for box in boxes:
          detections.append({
            'bbox': box.xyxy[0].cpu().numpy().tolist(),
            'confidence': float(box.conf[0]),
            'class': int(box.cls[0])
          })
      
      results_list.append({
        'image': img_path.name,
        'detections': detections,
        'num_detections': len(detections)
      })
    
    print(f"✅ Processed {len(image_files)} images")
    print(f"📁 Results saved to {output_dir}")
    
    return results_list


  def fine_tune_model(self, pretrained_path = None, epochs=50, lr=0.001):
    """
    Fine-tune an existing model with lower learning rate
    
    Args:
      pretrained_path: Path to pretrained model weights
      epochs: Number of fine-tuning epochs
      lr: Learning rate for fine-tuning
    """
    print(f"\n🔧 Fine-tuning model from {pretrained_path}...")
    
    self.pretrained_path = pretrained_path if pretrained_path is not None else f"{self.model_size}.pt"
    self.model = YOLO(self.pretrained_path)
    
    results = self.model.train(
      data=str(self.data_path / 'data.yaml'),
      epochs=epochs,
      imgsz=640,
      batch=16,
      lr0=lr,  # Initial learning rate
      lrf=0.01,  # Final learning rate factor
      save=True,
      project='pfm1_finetuning',
      name='exp',
      plots=True,
      device='cpu',
      resume=False,
    )
    
    print("✅ Fine-tuning complete!")
    return results
  
  def plot_training_results(self, training_dir='pfm1_finetuning/exp'):
    """Plot training metrics"""
    print("\n📊 Plotting training results...")
    
    results_csv = Path(training_dir) / 'results.csv'
    
    if not results_csv.exists():
      print(f"❌ Results file not found at {results_csv}")
      return
    
    # Read results
    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PFM-1 Detection Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: mAP metrics
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='green')
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='blue')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].set_title('Mean Average Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision and Recall
    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='orange')
    axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning rate
    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate (pg0)', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("✅ Training plots saved to 'training_results.png'")
    plt.show()

  
  def create_confusion_matrix(self, validation_results_path='runs/detect/val'):
    """Display confusion matrix from validation"""
    print("\n📊 Creating confusion matrix...")
    
    cm_path = Path(validation_results_path) / 'confusion_matrix.png'
    if cm_path.exists():
      img = plt.imread(cm_path)
      plt.figure(figsize=(10, 8))
      plt.imshow(img)
      plt.axis('off')
      plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
      plt.tight_layout()
      plt.show()
      print(f"✅ Confusion matrix displayed from {cm_path}")
    else:
      print(f"❌ Confusion matrix not found at {cm_path}")
    
    return None

  def analyze_detection_results(self, results_list):
    """Analyze detection results and create visualizations"""
    print("\n📊 Analyzing detection results...")
    
    # Extract statistics
    total_images = len(results_list)
    images_with_detections = sum(1 for r in results_list if r['num_detections'] > 0)
    total_detections = sum(r['num_detections'] for r in results_list)
    avg_detections = total_detections / total_images if total_images > 0 else 0
    
    # Confidence distribution
    all_confidences = []
    for r in results_list:
      for det in r['detections']:
        all_confidences.append(det['confidence'])
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('PFM-1 Detection Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Detection distribution
    detection_counts = [r['num_detections'] for r in results_list]
    axes[0].hist(detection_counts, bins=20, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Number of Detections per Image')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Detection Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence distribution
    if all_confidences:
      axes[1].hist(all_confidences, bins=30, color='green', edgecolor='black', alpha=0.7)
      axes[1].set_xlabel('Confidence Score')
      axes[1].set_ylabel('Frequency')
      axes[1].set_title('Confidence Distribution')
      axes[1].axvline(np.mean(all_confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_confidences):.3f}')
      axes[1].legend()
      axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Summary statistics
    axes[2].axis('off')
    mean_conf = np.mean(all_confidences) if all_confidences else 0.0
    median_conf = np.median(all_confidences) if all_confidences else 0.0
    min_conf = np.min(all_confidences) if all_confidences else 0.0
    max_conf = np.max(all_confidences) if all_confidences else 0.0

    summary_text = f"""
      Detection Summary
      ═══════════════════
      
      Total Images: {total_images}
      Images with Detections: {images_with_detections}
      Detection Rate: {images_with_detections/total_images*100:.1f}%
      
      Total Detections: {total_detections}
      Avg Detections/Image: {avg_detections:.2f}
      
      Confidence Stats:
      Mean: {mean_conf}
      Median: {median_conf}
      Min: {min_conf}
      Max: {max_conf}
    """
    axes[2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('detection_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Detection analysis saved to 'detection_analysis.png'")
    plt.show()
    
    return {
      'total_images': total_images,
      'images_with_detections': images_with_detections,
      'detection_rate': images_with_detections/total_images if total_images > 0 else 0,
      'total_detections': total_detections,
      'avg_detections_per_image': avg_detections,
      'confidence_stats': {
        'mean': float(np.mean(all_confidences)) if all_confidences else 0,
        'median': float(np.median(all_confidences)) if all_confidences else 0,
        'std': float(np.std(all_confidences)) if all_confidences else 0
      }
    }