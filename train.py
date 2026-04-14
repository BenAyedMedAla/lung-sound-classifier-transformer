import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix


from src.dataset import ASTDataset
from src.model import CustomAST
from src.sam import SAM

def train(args):
   
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Cihaz: {DEVICE}")
    
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    
    print(f"📥 Loading: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {args.data_path}. Lütfen preprocess.py çalıştırın.")

    data = np.load(args.data_path)
    X_train, y_train, d_train = data['X_train'], data['y_train'], data['device_train']
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']

    
    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    counts = np.bincount(y_train)
    weights = [1.0/counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    
    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True), 
        batch_size=args.batch_size, 
        sampler=sampler
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False), 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # 3. MODEL VE OPTIMIZER
    print("🧠 Preparing")
    model = CustomAST(num_classes=4).to(DEVICE)
    
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 4. EĞİTİM DÖNGÜSÜ
    print("🚀Train begins")
    best_score = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            
            criterion(model(inputs), labels).backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(DEVICE)
                
                logits = model(inputs) 
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        
        cm = confusion_matrix(all_labels, all_preds)
        se = np.sum(cm[1:, 1:]) / np.sum(cm[1:, :]) if np.sum(cm[1:, :]) > 0 else 0
        sp = cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) > 0 else 0
        score = (se + sp) / 2

        print(f"Epoch {epoch+1}: Avg Loss={running_loss/len(train_loader):.4f} | Score={score:.4f} (Se={se:.2f}, Sp={sp:.2f})")

        #
        if score > best_score:
            best_score = score
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    --> 💾 Last best Saved ({best_score:.4f})")

    print(f"\n🏆 Best Score: {best_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AST model with SAM for ICBHI")
    parser.add_argument("--data_path", type=str, default="./icbhi_ast_16k_8s_metadata.npz", help="Path to processed .npz file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    
    args = parser.parse_args()
    train(args)