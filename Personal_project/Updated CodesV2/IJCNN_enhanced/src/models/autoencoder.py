import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=10):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim), # Bottleneck
            nn.ReLU() 
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(
    X_train, 
    input_dim, 
    encoding_dim=10, 
    epochs=50, 
    batch_size=64, 
    lr=1e-3, 
    seed=42, 
    save_path="experiments/best_models/autoencoder.pth",
    verbose=False
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Autoencoder(input_dim, encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    start_time = time.time()
    history = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        
        epoch_loss /= len(X_tensor)
        history.append(epoch_loss)
        if verbose and (epoch+1) % 10 == 0:
            print(f"[Autoencoder] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
            
    train_time = time.time() - start_time
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    return model, history, train_time

def get_embeddings(model, X):
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        _, encoded = model(X_tensor)
    return encoded.cpu().numpy()
