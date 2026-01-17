import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from data.signal_generator import generate_dataset
from data.dataset import IQDataset
from src.model import InterferenceCNN

def main():
    X, y = generate_dataset(n_samples_per_class=500)

    dataset = IQDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = InterferenceCNN()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()

    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.2f} | Val Acc: {acc:.3f}"
        )

    torch.save(model.state_dict(), "results/interference_cnn.pth")
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
