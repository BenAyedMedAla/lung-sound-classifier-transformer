import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import ASTFeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import argparse
import gc

# --- MODÜLER İMPORTLAR ---
from src.dataset import ASTDataset
from src.model import CustomAST

def evaluate(args):
    # 1. AYARLAR
    gc.collect()
    torch.cuda.empty_cache()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Device: {DEVICE}")

    CLASSES = ['Normal', 'Crackle', 'Wheeze', 'Both']

    
    print(f"📦DataLoading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    data = np.load(args.data_path)
   
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']

    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    
    test_dataset = ASTDataset(X_test, y_test, d_test, processor, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    
    print(f"📦 Model yükleniyor: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {args.model_path}. Önce train.py çalıştırın.")

    model = CustomAST(num_classes=4).to(DEVICE)
    
   
    try:
        state_dict = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model ağırlıkları başarıyla yüklendi.")
    except Exception as e:
        print(f"⚠️ Hata: {e}")
        print("Model yüklenemedi. Dosya yolunu kontrol edin.")
        return

    model.eval()

    
    print("🔍 Test ediliyor...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels, _ in test_loader: 
            inputs = inputs.to(DEVICE)
            
            
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
                    preds = torch.argmax(logits, dim=1)
            else:
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    
    cm = confusion_matrix(all_targets, all_preds)

   
    se_numerator = np.sum(cm[1:, 1:])  
    se_denominator = np.sum(cm[1:, :]) 
    se = se_numerator / se_denominator if se_denominator > 0 else 0

  
    sp_numerator = cm[0, 0]           
    sp_denominator = np.sum(cm[0, :]) 
    sp = sp_numerator / sp_denominator if sp_denominator > 0 else 0


    score = (se + sp) / 2

    print(f"\n📊 Metrics:")
    print(f"   Sensitivity (Se): {se:.4f} ({se*100:.2f}%)")
    print(f"   Specificity (Sp): {sp:.4f} ({sp*100:.2f}%)")
    print(f"   Score:            {score:.4f} ({score*100:.2f}%)")

    if args.output_dir:
        print("\n🎨 Confusion Matrix Creating")
        os.makedirs(args.output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 7))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                         xticklabels=CLASSES, yticklabels=CLASSES,
                         annot_kws={"size": 14, "weight": "bold"},
                         cbar_kws={'label': 'Number of Samples'})

        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)

      
        metrics_text = f"Sensitivity (Se): {se*100:.2f}%  |  Specificity (Sp): {sp*100:.2f}%  |  Score: {score*100:.2f}%"
        plt.figtext(0.5, 0.02, metrics_text, wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        save_file = os.path.join(args.output_dir, "confusion_matrix.png")
        plt.savefig(save_file, dpi=600, bbox_inches='tight')
        print(f"✅ Görsel Kaydedildi: {save_file}")
        
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AST model for ICBHI")
    
    parser.add_argument("--data_path", type=str, default="./icbhi_ast_16k_8s_metadata.npz", help="Path to processed .npz file")
    
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth", help="Path to trained model (.pth)")
    
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save figures")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    
    args = parser.parse_args()
    evaluate(args)