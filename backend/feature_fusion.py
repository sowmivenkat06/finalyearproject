import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import copy
import json
from dataset import *
from feature_extractor import *

#############################################################
# Dataset
#############################################################

dataset224 = CervicalDataset(

    DATASET_PATH,

    transform_224

)

dataset299 = CervicalDataset(

    DATASET_PATH,

    transform_299

)

#############################################################
# Split Dataset
#############################################################

train_size = int(0.70 * len(dataset224))

val_size = int(0.15 * len(dataset224))

test_size = len(dataset224) - train_size - val_size

from torch.utils.data import Subset
import numpy as np

#############################################################
# Same indices for both datasets
#############################################################

np.random.seed(42)
indices = np.random.permutation(len(dataset224))

train_end = int(0.70 * len(indices))
val_end = int(0.85 * len(indices))

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

#############################################################
# 224 Dataset
#############################################################

train224 = Subset(dataset224, train_idx)
val224 = Subset(dataset224, val_idx)
test224 = Subset(dataset224, test_idx)

#############################################################
# 299 Dataset
#############################################################

train299 = Subset(dataset299, train_idx)
val299 = Subset(dataset299, val_idx)
test299 = Subset(dataset299, test_idx)

#############################################################
# DataLoader
#############################################################

train_loader224 = DataLoader(

    train224,

    batch_size=16,

    shuffle=False

)

train_loader299 = DataLoader(

    train299,

    batch_size=16,

    shuffle=False

)

val_loader224 = DataLoader(

    val224,

    batch_size=16,

    shuffle=False

)

val_loader299 = DataLoader(

    val299,

    batch_size=16,

    shuffle=False

)
#############################################################
# Attention Layer
#############################################################

class FeatureAttention(nn.Module):

    def __init__(self, feature_dim):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(feature_dim,2048),

            nn.ReLU(),

            nn.Linear(2048,feature_dim),

            nn.Sigmoid()

        )

    def forward(self,x):

        weights = self.attention(x)

        x = x * weights

        return x
#############################################################
# Fusion Classifier
#############################################################

class FusionClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.attention = FeatureAttention(17408)

        self.classifier = nn.Sequential(

            nn.BatchNorm1d(17408),

            nn.Linear(17408,4096),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(4096,1024),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(1024,512),

            nn.ReLU(),

            nn.Linear(512,NUM_CLASSES)

        )

    def forward(self,x):

        x = self.attention(x)

        x = self.classifier(x)

        return x
fusion_model = FusionClassifier().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    fusion_model.parameters(),

    lr=0.0001

)

from tqdm import tqdm
import copy

EPOCHS = 20
patience = 5

best_acc = 0
counter = 0

best_weights = copy.deepcopy(fusion_model.state_dict())

if __name__ == "__main__":

    ###################################################
    # Resume Training
    ###################################################

    os.makedirs("checkpoints", exist_ok=True)

    start_epoch = 0
    best_acc = 0
    counter = 0
    training_history = []

    checkpoint_files = []

    if os.path.exists("checkpoints"):

        checkpoint_files = [

            f for f in os.listdir("checkpoints")

            if f.endswith(".pth")

        ]

    if len(checkpoint_files) > 0:

        checkpoint_files.sort(

            key=lambda x: int(

                x.split("_")[-1].split(".")[0]

            )

        )

        latest = checkpoint_files[-1]

        print("Loading:", latest)

        checkpoint = torch.load(

            os.path.join("checkpoints", latest),

            map_location=DEVICE

        )

        fusion_model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

        start_epoch = checkpoint["epoch"]

        best_acc = checkpoint["best_accuracy"]

        counter = checkpoint["counter"]
        training_history = checkpoint.get("training_history", [])

        print(f"Resuming From Epoch {start_epoch+1}")

    else:

        print("Training From Scratch")

    ###################################################
    # Training Loop
    ###################################################

    for epoch in range(start_epoch, EPOCHS):

        fusion_model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        progress = tqdm(

            zip(train_loader224, train_loader299),

            total=len(train_loader224),

            desc=f"Epoch {epoch+1}/{EPOCHS}"

        )

        for (img224, label1), (img299, label2) in progress:

            img224 = img224.to(DEVICE)
            img299 = img299.to(DEVICE)

            labels = label1.to(DEVICE)

            optimizer.zero_grad()

            ###################################################
            # Extract Deep Features
            ###################################################

            features = extractor.extract(

                img224,

                img299

            )

            ###################################################
            # Fusion Network
            ###################################################

            outputs = fusion_model(features)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(outputs, 1)

            train_total += labels.size(0)

            train_correct += (pred == labels).sum().item()

            progress.set_postfix(

                Loss=f"{loss.item():.4f}",

                Acc=f"{100*train_correct/train_total:.2f}%"

            )

        train_acc = 100 * train_correct / train_total

        ###################################################
        # Validation
        ###################################################

        fusion_model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():

            for (img224, label1), (img299, label2) in zip(

                val_loader224,

                val_loader299

            ):

                img224 = img224.to(DEVICE)

                img299 = img299.to(DEVICE)

                labels = label1.to(DEVICE)

                features = extractor.extract(

                    img224,

                    img299

                )

                outputs = fusion_model(features)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, pred = torch.max(outputs, 1)

                val_total += labels.size(0)

                val_correct += (pred == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        training_history.append({"epoch": epoch + 1, "train_loss": train_loss / max(len(train_loader224), 1), "validation_loss": val_loss / max(len(val_loader224), 1), "train_accuracy": train_acc, "validation_accuracy": val_acc})

        print("\n")

        print("=" * 70)

        print(f"Epoch {epoch+1}/{EPOCHS}")

        print(f"Training Accuracy : {train_acc:.2f}%")

        print(f"Validation Accuracy : {val_acc:.2f}%")

        print("=" * 70)

        ###################################################
        # Save Best Model
        ###################################################

        if val_acc > best_acc:

            best_acc = val_acc

            best_weights = copy.deepcopy(

                fusion_model.state_dict()

            )

            torch.save(

                best_weights,

                "fusion/fusion_model.pth"

            )

            print("Best Fusion Model Saved")

            counter = 0

        else:

            counter += 1

            print(f"No Improvement ({counter}/{patience})")

        ###################################################
        # Save Checkpoint
        ###################################################

        checkpoint = {

            "epoch": epoch + 1,

            "model_state_dict": fusion_model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "best_accuracy": best_acc,

            "counter": counter
            ,"training_history": training_history

        }

        torch.save(

            checkpoint,

            f"checkpoints/checkpoint_epoch_{epoch+1}.pth"

        )

        os.makedirs("output", exist_ok=True)
        with open("output/training_history.json", "w", encoding="utf-8") as history_file:
            json.dump({"model": "Feature Fusion (Attention + MLP)", "epochs": training_history}, history_file, indent=2)

        print(f"Checkpoint Saved : Epoch {epoch+1}")

        if counter >= patience:

            print("\nEarly Stopping\n")

            break

    ###################################################
    # Load Best Model
    ###################################################

    fusion_model.load_state_dict(best_weights)

    print("\nFusion Training Completed")
