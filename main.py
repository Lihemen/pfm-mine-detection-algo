from detection_system import PFM1DetectionSystem

def main():
  print("=" * 30)
  print("PFM-1 MINE DETECTION SYSTEM")
  print("=" * 30)

  # Initialize system
  system = PFM1DetectionSystem(data_path='./data', model_size='yolov8n')

  # Step 1: Setup dataset (comment out when dataset is ready)
  print("\n" + "=" * 30)
  print("STEP 1: DATASET SETUP")
  print("=" * 30)
  system.setup_dataset_structure()
  system.create_data_yaml()


  # Step 2: Train model (uncomment when dataset is ready, comment out when training is done)
  print("\n" + "=" * 30)
  print("STEP 2: MODEL TRAINING")
  print("=" * 30)
  # training_results = system.train_model(epochs=100, imgsz=640, batch=16)  

  # Step 3: Fine-tune (optional - uncomment to use)
  print("\n" + "=" * 30)
  print("STEP 3: FINE-TUNING (OPTIONAL)")
  print("=" * 30)
  # system.fine_tune_model('training/exp/weights/best.pt', epochs=50)

  # Step 4: Load trained model
  print("\n" + "=" * 30)
  print("STEP 4: LOAD MODEL")
  print("=" * 30)
  # system.load_model('training/exp/weights/best.pt')
  
  # Step 5: Validate and calculate accuracy (depends on loaded trained model)
  print("\n" + "=" * 30)
  print("STEP 5: VALIDATION & ACCURACY")
  print("=" * 30)
  # validation_results = system.validate_model()
  
  # Step 6: Test on images (depends on loaded trained model)
  print("\n" + "=" * 30)
  print("STEP 6: TEST ON IMAGES")
  print("=" * 30)
  # test_results = system.test_on_images('data/images/test')
  
  # Step 7: Plot results  (depends on loaded trained model)
  print("\n" + "=" * 30)
  print("STEP 7: VISUALIZE RESULTS")
  print("=" * 30)
  # system.plot_training_results()
  # system.create_confusion_matrix()
  # analysis = system.analyze_detection_results(test_results)

if __name__ == "__main__":
  main()