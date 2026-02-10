import ssl
# Güvenlik sertifikası hatasını aşmak için:
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from src.data_loader import get_data_loaders
from src.model import KanserModel
from src.trainer import model_egit
import pandas as pd
import os
from glob import glob

# --- AYARLAR ---
CSV_YOLU = 'data/Data_Entry_2017.csv'
RESIM_KLASORU = 'data/images' 
EPOCH_SAYISI = 5 

def veri_hazirla():
    if os.path.exists('islenmis_veri.csv'):
        print("Hazır veri listesi yükleniyor...")
        return pd.read_csv('islenmis_veri.csv')

    print("Veri seti taranıyor...")
    df = pd.read_csv(CSV_YOLU)
    tum_resimler = glob(os.path.join(RESIM_KLASORU, '**', '*.png'), recursive=True)
    resim_map = {os.path.basename(x): x for x in tum_resimler}
    df['path'] = df['Image Index'].map(resim_map)
    df = df.dropna(subset=['path'])
    
    kanser_df = df[df['Finding Labels'].str.contains("Nodule|Mass")]
    saglikli_df = df[df['Finding Labels'] == "No Finding"]
    
    if len(saglikli_df) > len(kanser_df):
        saglikli_df = saglikli_df.sample(n=len(kanser_df), random_state=42)
        
    final_df = pd.concat([kanser_df, saglikli_df])
    final_df['label'] = final_df['Finding Labels'].apply(lambda x: 1 if ("Nodule" in x or "Mass" in x) else 0)
    final_df.to_csv('islenmis_veri.csv', index=False)
    return final_df

if __name__ == "__main__":
    print("--- 1. ADIM: Veri Hazırlığı ---")
    dataset_df = veri_hazirla()
    
    # Batch size 16 iyidir
    train_loader, test_loader = get_data_loaders(dataset_df, batch_size=16)
    
    print("\n--- 2. ADIM: Model Kurulumu ---")
    model = KanserModel()
    
    print("\n--- 3. ADIM: EĞİTİM BAŞLIYOR 🚀 ---")
    print("(Bu işlem bilgisayar hızına göre biraz sürebilir, sayılar akmaya başlayacak...)")
    
    # Modeli Eğit
    egitilmis_model = model_egit(model, train_loader, test_loader, epochs=EPOCH_SAYISI)
    
    # Kaydet
    torch.save(egitilmis_model.state_dict(), "kanser_tespit_modeli.pth")
    print("\n💾 Model başarıyla 'kanser_tespit_modeli.pth' olarak kaydedildi!")
    # açıklama satırı 