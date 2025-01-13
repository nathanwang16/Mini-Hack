

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt
from collections import defaultdict


num_epochs = 50
train_file = "/Users/xiaoyuwang/Desktop/Mini-Hack/2983_reversed_MPRA_categorized.txt"  # input data file
output_file = "12983_trained_model_ResNet_Decay.pth"  # output model file

#################################################
# 1. Device Selection (MPS on Apple Silicon, CUDA on Colab, else CPU)
#################################################
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# For reproducibility
SEED = 18
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

#################################################
# 2. Dataset Definition (With Family ID Parsing)
#################################################
class MPRADataset(Dataset):
    """
    Each line in the train_file:
      seqID   ACGTACGT... (295 bp)   L M H ... (295 tokens)

    We'll parse out:
      - family_id from seqID (e.g. seqID="family1_rev" => family_id="family1")
      - one-hot DNA
      - integer labels for L,M,H => 0,1,2
    """
    def __init__(self, txt_file):
        self.samples = []  # (dna_tensor, labels_tensor, family_id)
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()

                seq_id = parts[0]       # e.g. "family1_rev"
                dna_seq = parts[1]      # 295 bp
                labels_str = parts[2:]  # 295 tokens of L, M, H

                # Extract family_id by splitting seqID (adjust to your naming convention)
                family_id = seq_id.split('_')[0]

                # One-hot encode
                dna_tensor = self.dna_to_one_hot(dna_seq)

                # Convert L/M/H -> 0/1/2
                label_map = {'L': 0, 'M': 1, 'H': 2}
                labels_int = [label_map[lbl] for lbl in labels_str]
                labels_tensor = torch.tensor(labels_int, dtype=torch.long)

                self.samples.append((dna_tensor, labels_tensor, family_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def dna_to_one_hot(dna_seq):
        """
        A->0, C->1, G->2, T->3
        Returns shape (4, seq_len).
        """
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        one_hot = torch.zeros((4, len(dna_seq)), dtype=torch.float32)
        for i, base in enumerate(dna_seq):
            one_hot[mapping[base], i] = 1.0
        return one_hot

#################################################
# 3. Build the Full Dataset and Group by Family
#################################################
full_dataset = MPRADataset(train_file)
dataset_size = len(full_dataset)
print("Total samples in dataset:", dataset_size)

# Group sample indices by family_id
family_map = defaultdict(list)
for idx, (_, _, fam_id) in enumerate(full_dataset):
    family_map[fam_id].append(idx)

family_ids = list(family_map.keys())
random.shuffle(family_ids)

# Split family IDs into train vs. val (85/15)
train_family_count = int(0.80 * len(family_ids))
train_family_ids = family_ids[:train_family_count]
val_family_ids   = family_ids[train_family_count:]

# Flatten the actual indices
train_indices = []
for fid in train_family_ids:
    train_indices.extend(family_map[fid])

val_indices = []
for fid in val_family_ids:
    val_indices.extend(family_map[fid])

# Create Subsets
train_dataset = Subset(full_dataset, train_indices)
val_dataset   = Subset(full_dataset, val_indices)

print(f"Train families: {len(train_family_ids)}, Val families: {len(val_family_ids)}")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

#################################################
# 4. DataLoaders (Single-thread, no multi-threading)
#################################################
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

#################################################
# 5. CNN Model (3-class Output)
#################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(16)

        # 3 output channels => (L, M, H)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=3, kernel_size=1, padding=0)

    def forward(self, x):
        # x shape: (batch, 4, 295)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)  # => (batch, 3, 295)
        return x



class BasicBlock1D(nn.Module):
    """
    A 1D residual block with dropout:
    - conv1 -> bn1 -> relu -> dropout
    - conv2 -> bn2 -> [skip + identity] -> relu
    - optionally add dropout again if desired
    """
    expansion = 1

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=9, 
                 stride=1, 
                 padding=4, 
                 downsample=None, 
                 dropout=0.0):
        """
        dropout: probability of zeroing elements (0.0 means no dropout)
        """
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, 
                               padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Insert dropout
        self.drop1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, 
                               padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample
        self.dropout = dropout
        if dropout > 0:
            self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop1(out) if self.dropout > 0 else out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        # Optionally apply dropout again after addition+ReLU
        if self.dropout > 0:
            out = self.drop2(out)

        return out
   
   
def make_layer(block, in_channels, out_channels, blocks, kernel_size=9, stride=1, padding=4, dropout=0.0):
    """
    Constructs a sequence (nn.Sequential) of `blocks` BasicBlock1D.
    """
    layers = []
    downsample = None
    if in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
    # First block can handle 'downsample' if needed
    layers.append(block(in_channels, out_channels, kernel_size, stride, padding, downsample, dropout=dropout))
    
    # Additional blocks
    for _ in range(1, blocks):
        layers.append(block(out_channels, out_channels, kernel_size, stride, padding, dropout=dropout))
    
    return nn.Sequential(*layers)


class ResNet1D(nn.Module):
    """
    Simple 1D ResNet with dropout in each block.
    """
    def __init__(self, 
                 block=BasicBlock1D, 
                 layers=[2, 2], 
                 num_classes=3,
                 dropout=0.0):
        """
        layers: how many blocks per layer, e.g. [2,2] or [2,2,2,2]
        dropout: dropout probability
        """
        super(ResNet1D, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv1d(4, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # Example of 2-layers ResNet; adapt for more layers
        self.layer1 = make_layer(block, 64, 64, layers[0], dropout=dropout)
        self.layer2 = make_layer(block, 64, 128, layers[1], dropout=dropout)

        # Output conv => 3 classes
        self.conv_out = nn.Conv1d(128, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x shape: (batch, 4, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # => (batch,64,seq_len)
        x = self.layer2(x)  # => (batch,128,seq_len)

        x = self.conv_out(x)  # => (batch,3,seq_len)
        return x
   
   
model = ResNet1D(dropout = 0.2).to(device)

#################################################
# 6. Training Setup
#################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

train_losses = []
val_losses = []
train_corrects = []
val_corrects = []

def calc_correct(logits, targets):
    """
    logits: (batch, 3, 295)
    targets: (batch, 295) with values in {0,1,2}
    Returns fraction of correct predictions across all positions.
    """
    preds = logits.argmax(dim=1)  # => (batch, 295)
    correct = (preds == targets).sum().item()
    total = targets.numel()       # batch*295
    return correct / total

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    with torch.no_grad():
        for (dna_tensor, labels_tensor, _) in loader:
            dna_tensor = dna_tensor.to(device)
            labels_tensor = labels_tensor.to(device)

            logits = model(dna_tensor)          # => (batch, 3, 295)
            loss = criterion(logits, labels_tensor)
            total_loss += loss.item()

            total_corr += calc_correct(logits, labels_tensor)

    avg_loss = total_loss / len(loader)
    avg_corr = total_corr / len(loader)
    return avg_loss, avg_corr

#################################################
# 7. Training Loop
#################################################
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    total_train_corr = 0.0

    for (dna_tensor, labels_tensor, _) in train_loader:
        dna_tensor = dna_tensor.to(device)
        labels_tensor = labels_tensor.to(device)

        optimizer.zero_grad()
        logits = model(dna_tensor)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_corr += calc_correct(logits, labels_tensor)

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_corr = total_train_corr / len(train_loader)

    val_loss, val_corr = evaluate(model, val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    train_corrects.append(avg_train_corr)
    val_corrects.append(val_corr)
    
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} || "
          f"Train Acc: {avg_train_corr:.4f} | Val Acc: {val_corr:.4f}")

#################################################
# 8. Plotting
#################################################
def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Fraction Correct')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(train_losses, val_losses, train_corrects, val_corrects)

#################################################
# 9. Saving the Trained Model
#################################################
torch.save(model.state_dict(), output_file)
print(f"Model weights saved to {output_file}")