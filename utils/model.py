import torch
import torch.nn as nn

# -----------------------------
# Condition-aware Autoencoder
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=1028):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


# -----------------------------
# Model Loader
# -----------------------------
def load_trained_model(weight_path="autoencoder_weights.pth"):
    model = Autoencoder(input_dim=1028)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model
