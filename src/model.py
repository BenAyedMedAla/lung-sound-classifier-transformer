import torch.nn as nn
from transformers import ASTModel

class CustomAST(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # MIT'nin AudioSet ile eğittiği AST modeli
        # Bu model internetten otomatik indirilir (~350MB)
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        # Sınıflandırıcı Başlık (Dropout ekli)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        outputs = self.ast(x)
        # Sequence çıkışının ortalaması (Mean Pooling) - CLS token yerine daha stabil
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)
        return logits