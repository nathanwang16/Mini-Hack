import torch
import torch.nn as nn
import torch.nn.functional as F
import random

MODEL_PATH = "/Users/xiaoyuwang/Desktop/Mini-Hack/2983_trained_model_ResNet_Decay.pth"
OUTPUT_FILE = "2983predictions.txt"
###############################################
# 4. Parse Testing File & Predict
###############################################
TEST_FILE = "/Users/xiaoyuwang/Desktop/Mini-Hack/test_MPRA.txt"
# Format:  ID   295bp_sequence
# In total 7720 lines

###############################################
# 1. Define/Load Your Model Architecture
###############################################
# Make sure this matches exactly how you defined your model for training.
# For example, if you have a ResNet1D class with the same structure.

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

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, padding=4, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

def make_layer(block, in_channels, out_channels, blocks, kernel_size=9, stride=1, padding=4):
    layers = []
    downsample = None
    if in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
    layers.append(block(in_channels, out_channels, kernel_size, stride, padding, downsample))
    for _ in range(1, blocks):
        layers.append(block(out_channels, out_channels, kernel_size, stride, padding))
    return nn.Sequential(*layers)

class ResNet1D(nn.Module):
    def __init__(self, block=BasicBlock1D, layers=[2,2], num_classes=3):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(4, 64, kernel_size=9, stride=1, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = make_layer(block, 64, 64, layers[0])
        self.layer2 = make_layer(block, 64, 128, layers[1])

        self.conv_out = nn.Conv1d(128, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)   # => (batch,64,295)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # => (batch,64,295)
        x = self.layer2(x)  # => (batch,128,295)

        x = self.conv_out(x) # => (batch,3,295)
        return x

###############################################
# 2. Load the Model Weights
###############################################
# E.g., "trained_model.pth"


# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

model = ResNet1D().to(device)
#model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

###############################################
# 3. Utility to Convert DNA -> One-Hot
###############################################
def dna_to_one_hot(dna_seq):
    """
    A->0, C->1, G->2, T->3
    Return shape (4, len(dna_seq)) FloatTensor
    """
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    one_hot = torch.zeros((4, len(dna_seq)), dtype=torch.float32)
    for i, base in enumerate(dna_seq):
        idx = mapping[base]
        one_hot[idx, i] = 1.0
    return one_hot



# We'll store all "activating" predictions in one list, all "repressive" in another:
activating_positions = []  # will store (seqID, position)
repressive_positions = []  # will store (seqID, position)

num_L = 0
num_M = 0
num_H = 0

with open(TEST_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        seq_id  = parts[0]
        dna_seq = parts[1]
        # Convert to one-hot => shape (4,295)
        x = dna_to_one_hot(dna_seq).unsqueeze(0).to(device)  # => (1,4,295)

        # Predict
        with torch.no_grad():
            logits = model(x)           # => (1, 3, 295)
            preds = logits.argmax(dim=1) # => (1, 295)
            preds = preds.squeeze(0)     # => (295,)

        # For each position, check if 0(L), 1(M), or 2(H)
        for i, label in enumerate(preds.cpu().tolist()):
            pos_1_based = i + 1  # positions start at 1
            if label == 0:
                # L => "repressive"
                num_L += 1
                repressive_positions.append((seq_id, pos_1_based))
            elif label == 1:
                # M => "medium"
                num_M += 1
            else:
                # 2 => H => "activating"
                num_H += 1
                activating_positions.append((seq_id, pos_1_based))

###############################################
# 5. Summaries
###############################################
print(f"Total L (repressive) = {num_L}")
print(f"Total M (medium) = {num_M}")
print(f"Total H (activating) = {num_H}")

# user-specified thresholds
MAX_ACTIVATING = 100_000
MAX_REPRESSIVE = 50_000

# If we have more than 100k "H", randomly sample down:
if len(activating_positions) > MAX_ACTIVATING:
    activating_positions = random.sample(activating_positions, MAX_ACTIVATING)

# If we have more than 50k "L", randomly sample down:
if len(repressive_positions) > MAX_REPRESSIVE:
    repressive_positions = random.sample(repressive_positions, MAX_REPRESSIVE)

###############################################
# 6. Writing the Output
###############################################
# Format per line:
#   A or R, seqID, position
#   e.g.:  A seq123 101
# We'll output all "A" first, then all "R". Or you can interleave as you wish.


with open(OUTPUT_FILE, 'w') as out_f:
    # Activating
    for (seqID, pos) in activating_positions:
        out_line = f"A {seqID} {pos}\n"
        out_f.write(out_line)
    # Repressive
    for (seqID, pos) in repressive_positions:
        out_line = f"R {seqID} {pos}\n"
        out_f.write(out_line)

print(f"Done! Output saved to {OUTPUT_FILE}")
print(f"Final #Activating in output: {len(activating_positions)}")
print(f"Final #Repressive in output: {len(repressive_positions)}")