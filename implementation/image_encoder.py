import torch
import torch.nn as nn

class DinoV3ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, model_type: str = 'vitb16'):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov3', f'dinov3_{model_type}')
        #freeze les parametre et empêche le calcul de gradient
        for param in self.backbone.parameters():
            param.requires_grad = False

        dino_embed_dim = self.backbone.embed_dim
        #projection layer pour projeter dans la dimenstion voulue 
        self.projection = nn.Linear(dino_embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone.forward_features(x)
            #récupère les embeddings pour les images dans patches
            patches = outputs['x_norm_patchtokens'] # [B, N_patches, 768]
            '''récupère 4 embeddings spéciaux qui représentent des informations 
            générales les images et devrais aider dans les taches de prediction
            nous les mettrons et verons l'effet que sa fait avec et sans'''
            registers = outputs['x_norm_regtokens'] # [B, N_regs, 768]
            # Concatenate patches and registers along the sequence dimension
            # Memory shape: [B, N_patches + N_regs, 768]
            combined = torch.cat([patches, registers], dim=1)
            
        return self.projection(combined)