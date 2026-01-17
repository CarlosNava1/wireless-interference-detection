import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from data.signal_generator import generate_dataset
from data.dataset import IQDataset
from src.model import InterferenceCNN

def main():
    X, y = generate_dataset(n_samples_per_class=200)
    dataset = IQDataset(X, y)

    loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    model = InterferenceCNN()
    model.load_state_dict(torch.load("results/interference_cnn.pth"))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y_batch in loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
