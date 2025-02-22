import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, val_loader):
    model.eval()

    val_binary_preds, val_binary_labels = [], []
    val_multiclass_preds, val_multiclass_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            numerical_data, binary_labels, multiclass_labels = batch
            numerical_data = numerical_data.cuda()
            binary_labels = binary_labels.cuda()
            multiclass_labels = multiclass_labels.cuda()

            binary_logits, multiclass_logits = model(numerical_data)

            binary_preds = torch.sigmoid(binary_logits) > 0.5
            multiclass_preds = [torch.argmax(mc, dim=1) for mc in multiclass_logits]

            val_binary_preds.extend(binary_preds.cpu().numpy())
            val_binary_labels.extend(binary_labels.cpu().numpy())
            val_multiclass_preds.extend([pred.cpu().numpy() for pred in multiclass_preds])
            val_multiclass_labels.extend(multiclass_labels.cpu().numpy())

    binary_accuracy = accuracy_score(val_binary_labels, val_binary_preds)
    multiclass_accuracy = accuracy_score(val_multiclass_labels, val_multiclass_preds)

    print(f"Binary Classification Accuracy: {binary_accuracy:.4f}")
    print(f"Multiclass Classification Accuracy: {multiclass_accuracy:.4f}")
