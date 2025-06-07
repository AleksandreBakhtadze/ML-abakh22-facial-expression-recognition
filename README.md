# ML-abakh22-facial-expression-recognition

ეს რეპოზიტორია შეიცავს კოდს სხვადასხვა მიდგომებით Kaggle-ის  Facial Expression Recognition Challenge-სთვის,
რომელიც გულისხმობდა 48x48 ნაცრისფერი ფერის სურათების შვიდ ემოციად (გაბრაზებული, ზიზღი, შიში, ბედნიერი, სევდიანი, გაკვირვება, ნეიტრალური) კლასიფიცირებას.
ექსპერიმენტები რეგისტრირებულია Weights & Biases-ზე (Wandb) და გაშვებულია Google Colab-ში, GitHub-ზე ვერსიის კონტროლით.

Dataset: Kaggle FER2013 (~28,709 train, ~7,178 test images).

პირველი მიდგომა:

  მონაცემების დამუშავება:

    Grayscale(), RandomHorizontalFlip(p=0.5), RandomRotation(10).
    Resize(48,48), ToTensor(), Normalize((0.5,), (0.5,)).
  მოდელის არქიტექტურა:
    
      კონვოლუციური layer-ები:
          3 ბლოკი: Conv2d(1→32→64→128, 3x3, padding=1), ReLU, BatchNorm2d, MaxPool2d(2).
          Output: 128x6x6.
      FC Layers:  Dropout(0.5), Linear(128*6*6→256), ReLU, Dropout(0.3), Linear(256→7).
      Progressive filters, batch norm, and dropout for regularization (Lectures 15-16).

  Training და ევალუაცია
    
    Loss: CrossEntropyLoss
    გაუმჯობესებები:
      პირველი ექსპერიმენტი: Adam (lr=0.001).
      მეორე ექსპერიმენტი: SGD (lr=0.01, momentum=0.9).
      მეტრიცები: Accuracy, precision, recall, F1, confusion matrix (via torchmetrics).
  შედეგები:
  
  Adam:
  Val Accuracy: 58.69%, Val Loss: 1.0856.
  SGD + Momentum:
  Val Accuracy: 55.92%, Val Loss: 1.1461.

მეორე მიდგომა: 
  
  მონაცემთა დამუშავება:
    ToTensor(), Normalize(mean=[0.5], std=[0.5]).
    costum FERDataset ამუშავებს CSV პიქსელის მონაცემებს.
    DataLoader: Batch-ის ზომა 64.
  მოდელის არქიტექტურა
  კონვოლუციური layer-ები:
    3 ბლოკი: Conv2d(1→64→128→256, 3x3, padding=1), BatchNorm2d, ReLU, MaxPool2d(2).
    Output: 256x6x6.
    FC Layers: Dropout(0.5), Linear(256*6*6→512), ReLU, Dropout(0.5), Linear(512→7).
  Training და შეფასება:
    Loss: CrossEntropyLoss.
    გაუმჯობესებები:
      ექსპერიმენტი: Adam (lr=0.001).
      Overfitting ტესტი 20-ნიმუშიან ქვეჯგუფზე (მიაღწია 100% სიზუსტეს 14 ეპოქაში).
      მეტრიცები: Accuracy, confusion matrix, precision, recall, F1.
  შედეგები:
    Adam:
    Val Accuracy: 54,82%, Val Loss: 54,9784.

მესამე მიდგომა: ImprovedCNN with 
  მონაცემთა დამუშავება:
    Train:
    Rotate(limit=15, p=0.5), HorizontalFlip(p=0.5), RandomBrightnessContrast(p=0.3).
    Normalize(mean=[0.5], std=[0.5]), ToTensorV2.
    costum FERDataset ამუშავებს CSV პიქსელის მონაცემებს.
    Val: 
    Normalize(mean=[0.5], std=[0.5]), ToTensorV2.
    DataLoader: Batch size 128, shuffle train-სთვის.
  მოდელის არქიტექტურა
    კონვოლუციური layer-ები:
    3 ბლოკი: 2x Conv2d(1→64, 64→128, 128→256, 3x3, padding=1), BatchNorm2d, ReLU, followed by MaxPool2d(2).
    Output: 256x6x6.
    FC Layers: Dropout(0.4), Linear(256*6*6→1024), ReLU, Dropout(0.4), Linear(1024→512), ReLU, Dropout(0.4), Linear(512→7).
  Training და შეფასება:
    Loss: CrossEntropyLoss.
  გაუმჯობესებები:
    ექსპერიმენტი: Adam (lr=0.001), 30 epochs.
    Overfitting test: [Pending, recommend balanced 20-sample subset with minimal transforms].
    მეტრიცები: Accuracy, precision, recall, F1, confusion matrix (via sklearn).
  შედეგები 
  Adam:
    Val Accuracy: 59.46%, Val Loss: 1.0893.
    Train Accuracy: 57.92%.

მეოთხე მიდგომა: Basic CNN, ResNet, and Vision Transformer
ამ მიდგომაში ერთ ნოუთბუქში მქონდა სამი მოდელი 

პირველი ექპერიმენტი: Basic CNN
  მონაცემთა დამუშავება:
    Train:
    albumentations: HorizontalFlip(p=0.5), ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5), RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), GaussNoise(var_limit=(10.0, 50.0), p=0.3),                                        CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3).
    Normalize(mean=[0.485], std=[0.229]), ToTensorV2.
    costum FERDataset ამუშავებს CSV პიქსელის მონაცემებს.
    90-10 stratified train-validation გაყოფა.
    WeightedRandomSampler დაუბალანლების გამოსასწორებლად.  
    Val: 
    Normalize(mean=[0.485], std=[0.229]), ToTensorV2.
    DataLoader: Batch size 128 (train), 32 (val), sampler train-ისთვის.
  მოდელის არქიტექტურა:
    კონვოლუციური layer-ები:
    3 ბლოკი: Conv2d(1→32, 32→64, 64→128, 3x3, padding=1), BatchNorm2d, ReLU, MaxPool2d(2), Dropout(0.25).
    Output: 128x6x6.
    FC Layers: Linear(128*6*6→256), BatchNorm1d, ReLU, Dropout(0.5), Linear(256→7).
  Training და შეფასება:
    Loss: Focal Loss (gamma=2.0, weighted for imbalance).
  გაუმჯობესებეი:
    Experiment: AdamW (lr=0.001, weight_decay=1e-4), 50 epoch, CosineAnnealingWarmRestarts(T_0=10).
    Gradient clipping (max_norm=1.0).
    Early stopping (patience=10).
    Overfitting test 
  მეტრიცები: Accuracy, macro precision, recall, F1, confusion matrix (via torchmetrics).
  შედეგები:
    AdamW:
      Val Accuracy: 58.13%, Val Loss: 0.5970.
      Train Accuracy: 62.31%.
      Macro F1: 0.5554.
  დასკვნა: WeightedRandomSampler-მა F1 score გააუმჯობესა, მაგრამ გვაქვს overfit რადგან ტრაინ გაცილებით უკეთეს შედეგს იძლევა ვიდრე ტესტი
მეორე ექსპერიმენტი: ResNet
  მონაცემების დამუშავება 
    Train:
    albumentations: HorizontalFlip(p=0.5), ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5), RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), GaussNoise(var_limit=(10.0, 50.0), p=0.3),                                        CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3).
    Normalize(mean=[0.485], std=[0.229]), ToTensorV2.
    costum FERDataset ამუშავებს CSV პიქსელის მონაცემებს.
    90-10 stratified train-validation გაყოფა.
    WeightedRandomSampler დაუბალანლების გამოსასწორებლად.  
    Val: 
    Normalize(mean=[0.485], std=[0.229]), ToTensorV2.
    DataLoader: Batch size 128 (train), 32 (val), sampler train-ისთვის.
  მოდელის არქიტექტურა:
    კონვოლუციური layer-ები:
      Conv2d(1→32, 3x3, padding=1), BatchNorm2d, ReLU.
      სამი ლეიერი 2 residual blocks(32→32, 32→64, 64→128).
    Residual block: 2x Conv2d(3x3), BatchNorm2d, ReLU, shortcut connection.
                    AdaptiveAvgPool2d(1,1).
    FC Layers: Linear(128→7).
  Training და შეფასება:
    Loss: Focal Loss (gamma=2.0, weighted for imbalance).
  გაუმჯობესებები:
    Experiment: AdamW (lr=0.001, weight_decay=1e-4), 50 epoch (stopped at 48), CosineAnnealingWarmRestarts(T_0=10).
    Gradient clipping (max_norm=1.0).
    Early stopping (patience=10).
    Overfitting test 
  მეტრიცები: Accuracy, macro precision, recall, F1, confusion matrix (via torchmetrics).
  შედეგები
    AdamW:
      Val Accuracy: 63.95%, Val Loss: 0.6513.
      Train Accuracy: 80.78%.
      Macro F1: 0.6278.
  დასკვნა: **საუკეთესო** შედეგი , დაბალანსირებული მეტრიცებ, გაუმჯობესებული F1 score, მაგრამ აქაც შეინისნება overfit    

მესამე ექსპერიმენტი: Vision Transformer (ViT)
  მონაცემების დამუშავება:
    Train:
    albumentations: HorizontalFlip(p=0.5), ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5), RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), GaussNoise(var_limit=(10.0, 50.0), p=0.3),                                        CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3).
    Normalize(mean=[0.485], std=[0.229]), ToTensorV2.
    costum FERDataset ამუშავებს CSV პიქსელის მონაცემებს.
    90-10 stratified train-validation გაყოფა.
    WeightedRandomSampler დაუბალანლების გამოსასწორებლად.  
    Val: 
    Normalize(mean=[0.485], std=[0.229]), ToTensorV2.
    DataLoader: Batch size 128 (train), 32 (val), sampler train-ისთვის.
  მოდელის არქიტექტურა:
    კონვოლუციური layer-ები:
        Patch Embedding:  48x48 images დავყავით 8x8 patche-ებად.
        Linear(64→64) patchembedding-ისთვის.
    Transformer:
    4 encoder ლეიერი, 4 heads, dim=64.
    CLS token + positional embeddings.
    FC Layers: Linear(64→32), ReLU, Linear(32→7).
Training და ევალუაცია:
  Loss: Focal Loss (gamma=2.0, weighted for imbalance).
გაუმჯობესებები:
  Experiment: AdamW (lr=0.0005, weight_decay=1e-4), 50 epoch (stopped at 11), CosineAnnealingWarmRestarts(T_0=10).
  Gradient clipping (max_norm=1.0).
  Early stopping (patience=10).
  Overfitting test
მეტრიცები: Accuracy, macro precision, recall, F1, confusion matrix (via torchmetrics).
შედეგები
  AdamW:
  Val Accuracy: 1.53%, Val Loss: Unknown (early stopping).
  Train Accuracy: ~14%.
  Macro F1: 0.0043.
დასკვნა: ყველაზე ცუდი შედეგი , Early stopping-მა შეაჩერა.


