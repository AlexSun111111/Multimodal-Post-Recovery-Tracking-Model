import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from model_text import TextModel


def train_text_model(model, train_loader, val_loader, epochs=200, lr=1e-4, batch_size=16):
    # Loss functions
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer and Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids, attention_mask, numerical_data, binary_labels, multiclass_labels = batch
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            numerical_data = numerical_data.cuda()
            binary_labels = binary_labels.cuda()
            multiclass_labels = multiclass_labels.cuda()

            optimizer.zero_grad()

            binary_logits, multiclass_logits = model(input_ids, attention_mask)

            # Binary classification loss
            binary_loss = bce_loss_fn(binary_logits, binary_labels)

            # Multiclass classification loss
            multiclass_loss = sum(ce_loss_fn(mc, multiclass_labels[:, i]) for i, mc in enumerate(multiclass_logits))

            loss = binary_loss + multiclass_loss
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_binary_preds, val_binary_labels = [], []
        val_multiclass_preds, val_multiclass_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, numerical_data, binary_labels, multiclass_labels = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                numerical_data = numerical_data.cuda()
                binary_labels = binary_labels.cuda()
                multiclass_labels = multiclass_labels.cuda()

                binary_logits, multiclass_logits = model(input_ids, attention_mask)

                binary_loss = bce_loss_fn(binary_logits, binary_labels)
                multiclass_loss = sum(ce_loss_fn(mc, multiclass_labels[:, i]) for i, mc in enumerate(multiclass_logits))

                loss = binary_loss + multiclass_loss
                val_loss += loss.item()

                # Collect predictions for metrics calculation
                binary_preds = torch.sigmoid(binary_logits) > 0.5
                multiclass_preds = [torch.argmax(mc, dim=1) for mc in multiclass_logits]

                val_binary_preds.extend(binary_preds.cpu().numpy())
                val_binary_labels.extend(binary_labels.cpu().numpy())
                val_multiclass_preds.extend([pred.cpu().numpy() for pred in multiclass_preds])
                val_multiclass_labels.extend(multiclass_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Learning Rate Adjustment
        scheduler.step(val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
