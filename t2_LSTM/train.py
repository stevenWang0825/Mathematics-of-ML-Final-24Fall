import torch
from tqdm.auto import tqdm

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = correct / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = correct / total_samples
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in tqdm(range(num_epochs)):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        # 打印或记录当前 epoch 的结果
        if epoch%100==0:
            print(f"Epoch[{epoch+1}/{num_epochs}] "
                f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
                f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f}")

    return train_losses, train_accs, val_losses, val_accs