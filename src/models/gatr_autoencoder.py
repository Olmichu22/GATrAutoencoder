import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from .gatr_module import GATrBasicModule


class GATrAutoencoder(nn.Module):
    """
    Autoencoder that uses GATr blocks for encoding and decoding.

    Expected inputs:
      - mv_v_part: (N, 3) point or vector
      - mv_s_part: (N, 1) geometric scalar
      - scalars: (N, F_in)
      - batch: (N,) batch indices
    """

    def __init__(
        self,
        cfg_enc,
        cfg_dec,
        cfg_agg=None,
        latent_s_channels=16,
    ):
        super().__init__()

        # Encoder: maps inputs -> latent scalars (and latent mv if needed)
        self.encoder = GATrBasicModule(
            hidden_mv_channels=cfg_enc["hidden_mv_channels"],
            hidden_s_channels=cfg_enc["hidden_s_channels"],
            num_blocks=cfg_enc["num_blocks"],
            in_s_channels=cfg_enc["in_s_channels"],
            in_mv_channels=cfg_enc["in_mv_channels"],
            out_mv_channels=cfg_enc["out_mv_channels"],
            dropout=cfg_enc["dropout"],
            out_s_channels=cfg_enc["out_s_channels"]
        )
      
        self.compressor = nn.Linear(cfg_enc["out_s_channels"] + cfg_enc["out_mv_channels"], latent_s_channels) # Compress concatenated latent mv and scalar to a smaller latent space
        self.coord_projector = nn.Linear(latent_s_channels + cfg_enc["out_s_channels"] + 16, 4) # Project latent mv to the same dimension as latent scalars for aggregation
        self.scalar_projector = nn.Linear(latent_s_channels + cfg_enc["out_s_channels"] + 16, cfg_dec["in_s_channels"]) # Project latent scalars to the same dimension as input scalars for decoding
        # Decoder: maps latent scalars (+ optional mv) -> reconstructed scalars
        self.decoder = GATrBasicModule(
            hidden_mv_channels=cfg_dec["hidden_mv_channels"],
            hidden_s_channels=cfg_dec["hidden_s_channels"],
            num_blocks=cfg_dec["num_blocks"],
            in_s_channels=cfg_dec["in_s_channels"],
            in_mv_channels=cfg_dec["in_mv_channels"],
            out_mv_channels=cfg_dec["out_mv_channels"],
            dropout=cfg_dec["dropout"],
            out_s_channels=cfg_dec["out_s_channels"],
        )

    def encode(self, mv_v_part, mv_s_part, scalars, batch):
        mv_latent, s_latent, point_latent, scalar_latent = self.encoder(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            batch=batch,
        )
        return mv_latent, s_latent, point_latent, scalar_latent

    def decode(self, mv_latent, mv_s_part, s_latent, batch):
        mv_rec, s_rec, point_rec, scalar_rec = self.decoder(
            mv_v_part=mv_latent.squeeze(1),
            mv_s_part=mv_s_part,
            scalars=s_latent,
            batch=batch,
        )
        return mv_rec, s_rec, point_rec, scalar_rec

    def forward(self, mv_v_part, mv_s_part, scalars, batch):
        mv_latent, s_latent, point_latent, scalar_latent = self.encode(
            mv_v_part, mv_s_part, scalars, batch
        ) # encode the data
        
        
        latent_concat = torch.cat([mv_latent.squeeze(1), s_latent], dim=-1) # (N, 16 + F_s)
        latent_compressed = self.compressor(latent_concat) # (N, latent_s_channels)
        # Compresse to very small latent space
        
        # aggregate event information by taking mean of latent representations for each event
        mv_latent_agg = scatter_mean(mv_latent.squeeze(1), batch, dim=0) # (B, 16)
        s_latent_agg = scatter_mean(s_latent, batch, dim=0) # (B, F_s)
        
        # expand mv_latent_agg and s_latent_agg to (N, 16) and (N, F_s) respectively for concatenation
        mv_latent_agg_expanded = mv_latent_agg[batch] # (N, 16)
        s_latent_agg_expanded = s_latent_agg[batch] # (N, F_s)
        
        # create full latent rpr by concatenating compressed latent, aggregated scalar latent, and aggregated mv latent
        latent_full_repr = torch.cat([latent_compressed, s_latent_agg_expanded, mv_latent_agg_expanded], dim=-1) # (B, latent_s_channels + F_s + 16)
        
        # project to 3D coordinates for decoding  
        mv_latent_agg_projected = self.coord_projector(latent_full_repr, dim=-1)
        
        # project to s_in dimension for decoding  
        s_latent_agg_projected = self.scalar_projector(latent_full_repr, dim=-1)
        
        
        mv_rec, s_rec, point_rec, scalar_rec = self.decode(
            mv_latent_agg_projected[:, :3], mv_latent_agg_projected[:, 3:], s_latent_agg_projected, batch
        )

        return {
            "mv_latent": mv_latent,
            "s_latent": s_latent,
            "point_latent": point_latent,
            "scalar_latent": scalar_latent,
            "mv_rec": mv_rec,
            "s_rec": s_rec, # the rest of the variables of the hit
            "point_rec": point_rec, # coordinate of the detector
            "scalar_rec": scalar_rec, # layer of the detector (for example)
        }
