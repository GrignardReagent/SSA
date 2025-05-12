from sklearn.metrics import accuracy_score
import torch

def evaluate_model(model, data_loader, output_size):
    """
    Evaluate a trained Classifier on a given DataLoader.

    Args:
        model (nn.Module): Trained model with .eval() and .to(device) set.
        data_loader (DataLoader): PyTorch DataLoader for evaluation (e.g., test_loader).
        output_size (int): Number of classes (1 for binary classification).

    Returns:
        float: Accuracy on the dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(model.device)
            y_batch = y_batch.to(model.device)

            outputs = model(X_batch).squeeze()

            if output_size == 1:
                preds = (torch.sigmoid(outputs) > 0.5).int()
            else:
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc

