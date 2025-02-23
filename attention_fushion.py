import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalAttention(nn.Module):
    """Multi-modal Attention Fusion Module
    Args:
        embed_dim (int): Embedding dimension of each modality
        num_modalities (int): Number of input modalities (default: 3 for text/numeric/cinematic)
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, embed_dim=256, num_modalities=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities

        # Cross-modal attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Dynamic weight generator
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_modalities),
            nn.Softmax(dim=-1)
        )

        # Weight recording buffers
        self.register_buffer('text_weights', torch.tensor([]))
        self.register_buffer('numeric_weights', torch.tensor([]))
        self.register_buffer('cinematic_weights', torch.tensor([]))

    def forward(self, modalities):
        """
        Args:
            modalities (List[torch.Tensor]): List of modality features
                [text_feat (B, D), numeric_feat (B, D), cinematic_feat (B, D)]
        Returns:
            torch.Tensor: Fused feature tensor (B, D)
            torch.Tensor: Modality weights (B, M)
        """
        # Stack modalities for batch processing
        combined = torch.stack(modalities, dim=1)  # (B, M, D)
        batch_size = combined.size(0)

        # Generate dynamic weights
        weights = self.weight_net(combined.view(batch_size, -1))  # (B, M)

        # Record weights for analysis
        self._update_weight_buffers(weights)

        # Cross-modal attention using text as query
        attn_output, _ = self.cross_attn(
            query=combined[:, 0].unsqueeze(1),  # Text features as query
            key=combined,
            value=combined
        )

        # Weighted fusion with residual connection
        weighted_features = torch.sum(weights.unsqueeze(-1) * combined, dim=1)
        fused_feature = weighted_features + attn_output.squeeze(1)

        return fused_feature, weights

    def _update_weight_buffers(self, weights):
        """Update registered buffers with current batch weights"""
        with torch.no_grad():
            self.text_weights = torch.cat([self.text_weights, weights[:, 0].cpu()])
            self.numeric_weights = torch.cat([self.numeric_weights, weights[:, 1].cpu()])
            self.cinematic_weights = torch.cat([self.cinematic_weights, weights[:, 2].cpu()])


class HierarchicalFeatureFuser(nn.Module):
    """Hierarchical Feature Fusion Module
    Args:
        embed_dim (int): Feature dimension
        expansion_ratio (int): MLP expansion ratio
        dropout (float): Dropout probability
    """

    def __init__(self, embed_dim=256, expansion_ratio=4, dropout=0.1):
        super().__init__()
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion_ratio, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.adaptive_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (B, D)
        Returns:
            torch.Tensor: Enhanced features (B, D)
        """
        # Channel attention
        attn_out, _ = self.channel_attn(
            x.unsqueeze(1),  # Add sequence dimension
            x.unsqueeze(1),
            x.unsqueeze(1)
        )

        # MLP processing
        mlp_out = self.feature_mlp(x)

        # Adaptive fusion
        return self.adaptive_alpha * attn_out.squeeze(1) + (1 - self.adaptive_alpha) * mlp_out