import torch
import torch.nn as nn
from torchvision import models

class KanserModel(nn.Module):
    def __init__(self):
        super(KanserModel, self).__init__()
        
        # --- BURAYA DİKKAT: Artık DenseNet var ---
        print("🧠 DenseNet121 (X-Ray Uzmanı) indiriliyor...") 
        
        # ResNet yerine DenseNet121 kullanıyoruz
        self.densenet = models.densenet121(weights='DEFAULT')
        
        # DenseNet'in özellik sayısını al (Genelde 1024)
        num_features = self.densenet.classifier.in_features
        
        # Son katmanı değiştiriyoruz
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        return self.densenet(x)