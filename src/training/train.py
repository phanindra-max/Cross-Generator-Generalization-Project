"""
Training script for deepfake detection models.

Supports training both the constrained CNN and baseline ResNet with configurable
hyperparameters, logging, and checkpoint saving.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.constrained_cnn import ConstrainedCNN
from src.models.baseline_resnet import BaselineResNet
from src.data.faceforensics_loader import get_dataloader
from src.evaluation.stratified_metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--model", choices=["constrained_cnn", "baseline_resnet"],
                        default="constrained_cnn")
    parser.add_argument("--dataset", default="faceforensics")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--manipulation", default="Deepfakes",
                        choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
    parser.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def get_model(model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name == "constrained_cnn":
        return ConstrainedCNN(num_classes=num_classes)
    elif model_name == "baseline_resnet":
        return BaselineResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    model = get_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = get_dataloader(
        root_dir=args.data_root, split="train",
        manipulation=args.manipulation, compression=args.compression,
        batch_size=args.batch_size,
    )
    val_loader = get_dataloader(
        root_dir=args.data_root, split="val",
        manipulation=args.manipulation, compression=args.compression,
        batch_size=args.batch_size,
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("AUC/val", val_metrics["auc"], epoch)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "args": vars(args),
            }, checkpoint_dir / "best.pth")
            print(f"  -> New best model saved (AUC: {best_auc:.4f})")

    writer.close()
    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
