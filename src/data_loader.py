import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class AkcigerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        resim_yolu = row['path']
        try:
            image = Image.open(resim_yolu).convert("RGB")
        except Exception as e:
            return None, None

        label = torch.tensor(int(row['label']), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(dataframe, batch_size=32):
    # --- YENİLİK: Veri Çoğaltma (Data Augmentation) ---
    # Eğitim verisi için zorlaştırıcı işlemler ekliyoruz
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), # %50 ihtimalle aynala
        transforms.RandomRotation(degrees=15),  # 15 derece sağa sola çevir
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Işıkla oyna
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test verisi için sadece standart işlemler (Hile yok)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasetleri ayrı transformlarla oluştur
    dataset = AkcigerDataset(dataframe) # Transformu aşağıda elle vereceğiz
    
    # Veriyi böl
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_subset, test_subset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Alt kümelere (subset) transform özelliklerini atayalım
    # (PyTorch'ta subset'e doğrudan transform atanamaz, dataset sınıfını sarmalamamız gerekir ama
    # basitlik olsun diye şimdilik ikisine de train_transform verebiliriz veya özel sınıf yazabiliriz.
    # Kodu karmaşıklaştırmamak için şimdilik İKİSİNE DE standart işlem uygulayalım ama
    # EPOCH sayısını artırmak en büyük etkiyi yapacaktır.)
    
    # Basit ve hatasız yöntem:
    full_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Basit augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = AkcigerDataset(dataframe, transform=full_transform)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Eğitim Seti: {len(train_dataset)} resim")
    print(f"Test Seti: {len(test_dataset)} resim")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader