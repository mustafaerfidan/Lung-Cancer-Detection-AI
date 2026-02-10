import torch
import torch.nn as nn
import torch.optim as optim

def model_egit(model, train_loader, test_loader, epochs=10): # Epoch'u varsayılan 10 yaptık
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Eğitim şu cihazda yapılacak: {device}")
    
    model = model.to(device)
    
    # 1. POS_WEIGHT: Kanserli veriyi kaçırmanın cezası daha büyük olsun
    # (Eğer verin dengeliyse bu şart değil ama yine de iyidir)
    pos_weight = torch.tensor([1.5]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    # 2. Learning Rate'i biraz daha düşük başlatalım (Daha hassas)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # --- YENİLİK: SCHEDULER (Vites Kutusu) ---
    # Eğer 'val_loss' (test hatası) 2 epoch boyunca düşmezse, öğrenme hızını 10 kat azalt.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    best_acc = 0.0 # En iyi başarıyı takip edelim
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} Başlıyor ---")
        
        # --- EĞİTİM ---
        model.train()
        toplam_loss = 0
        dogru_tahmin = 0
        toplam_veri = 0
        
        for batch_idx, (resimler, etiketler) in enumerate(train_loader):
            resimler, etiketler = resimler.to(device), etiketler.to(device)
            etiketler = etiketler.unsqueeze(1)
            
            cikti = model(resimler)
            loss = criterion(cikti, etiketler)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            toplam_loss += loss.item()
            tahminler = torch.sigmoid(cikti) > 0.5
            dogru_tahmin += (tahminler == etiketler).sum().item()
            toplam_veri += etiketler.size(0)
            
            if batch_idx % 100 == 0:
                print(f"   Adım {batch_idx}: Loss: {loss.item():.4f}")
        
        egitim_basarisi = 100 * dogru_tahmin / toplam_veri
        print(f"📉 Eğitim Ort. Loss: {toplam_loss/len(train_loader):.4f} | Başarı: %{egitim_basarisi:.2f}")
        
        # --- TEST ---
        test_loss, test_acc = model_test_et(model, test_loader, device, criterion)
        
        # Scheduler'a haber ver: "Bak bakalım hata düştü mü?"
        scheduler.step(test_loss)
        
        # En iyi modeli kaydet (Her epoch sonunda değil, sadece rekor kırınca)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "en_iyi_kanser_modeli.pth")
            print(f"⭐ YENİ REKOR! Model kaydedildi. (%{best_acc:.2f})")

    print(f"\n🎉 EĞİTİM BİTTİ! En yüksek başarı: %{best_acc:.2f}")
    return model

def model_test_et(model, test_loader, device, criterion):
    model.eval()
    toplam_loss = 0
    dogru = 0
    toplam = 0
    
    with torch.no_grad():
        for resimler, etiketler in test_loader:
            resimler, etiketler = resimler.to(device), etiketler.to(device)
            etiketler = etiketler.unsqueeze(1)
            
            cikti = model(resimler)
            loss = criterion(cikti, etiketler) # Test loss'u da hesaplayalım ki scheduler bilsin
            toplam_loss += loss.item()
            
            tahminler = torch.sigmoid(cikti) > 0.5
            dogru += (tahminler == etiketler).sum().item()
            toplam += etiketler.size(0)
            
    avg_loss = toplam_loss / len(test_loader)
    acc = 100 * dogru / toplam
    print(f"🧪 Test Sonucu -> Loss: {avg_loss:.4f} | Başarı: %{acc:.2f}")
    return avg_loss, acc