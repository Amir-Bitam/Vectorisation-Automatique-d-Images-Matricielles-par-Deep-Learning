import torch
import torch.nn as nn

class PathDecoder(nn.Module):
    def __init__(
        self, 
        num_paths: int = 128, 
        hidden_dim: int = 256, 
        nheads: int = 8, 
        num_layers: int = 6
    ):
        """
        Args:
            num_paths: How many SVG paths the model can "draw".
            hidden_dim: The D dimension matching the Image Encoder.
            nheads: Number of attention heads.
            num_layers: Number of Transformer Decoder layers.
        """
        super().__init__()
        
        # 1. Learnable Path Queries
        # These are the "seeds" that will become SVG paths.
        self.path_queries = nn.Embedding(num_paths, hidden_dim)
        
        # 2. Transformer Decoder
        # Standard implementation: Includes Self-Attention, Cross-Attention, and FFN.

        '''ici on défini un layer de transformer decoder qui va être stack par le transformer decoder,
        par défault 6 d'entre eux seront stack, chaque layer reçoit la sortie de l'encoder et 
        réfléchit sur quel paths choisir pour reconstruire l'image'''
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=nheads, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: Output from DinoV3ImageEncoder [B, N_src, D]
            
        Returns:
            Path embeddings: [B, num_paths, D]
        """
        batch_size = image_features.shape[0]
        
        # Expand path queries to match batch size: [num_paths, D] -> [B, num_paths, D]
        # We use .weight to get the actual tensor from the nn.Embedding layer.
        queries = self.path_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # The Decoder:
        # tgt = Queries (Q)
        # memory = Image Features (K, V)
        path_embeddings = self.decoder(tgt=queries, memory=image_features)
        
        return path_embeddings

class MLP(nn.Module):
    """A simple 3-layer MLP used for the prediction heads."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid() # Bounds output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SVGPredictionHeads(nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 256, 
        num_points_per_path: int = 4 # e.g., 4 points for a cubic Bezier curve
    ):
        super().__init__()
        
        # 1. Coordinate Head
        # Output: [B, N_paths, num_points * 2] (multiplied by 2 for x and y)
        self.coord_head = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=num_points_per_path * 2)
        
        # 2. Color Head
        # Output: [B, N_paths, 3] (R, G, B)
        self.color_head = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=3)
        
        # 3. Opacity (Alpha) Head
        # Output: [B, N_paths, 1]
        self.alpha_head = MLP(in_dim=hidden_dim, hidden_dim=hidden_dim // 2, out_dim=1)

    def forward(self, path_embeddings: torch.Tensor):
        """
        Args:
            path_embeddings: Output from PathDecoder [B, N_paths, D]
        Returns:
            Dictionary containing coordinates, colors, and opacities.
        """
        coords = self.coord_head(path_embeddings)
        colors = self.color_head(path_embeddings)
        alphas = self.alpha_head(path_embeddings)
        
        return {
            "coords": coords,
            "colors": colors,
            "alphas": alphas
        }