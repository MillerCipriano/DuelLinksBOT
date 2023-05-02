import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def train_button_detector(model, train_loader, val_loader, num_epochs=20, save_path='button_detector.pth'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        for phase in ['treino', 'validacao']:
            if phase == 'treino':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'treino'):
                    outputs = model(inputs)
                    loss = criterion(F.log_softmax(outputs, dim=1)[:, 1], labels.float())

                    preds = (outputs.view(-1) > 0).long()

                    if phase == 'treino':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(labels.size(0), -1).argmax(1) == labels)


            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    torch.save(model.state_dict(), save_path)
    print(f'Modelo salvo em {save_path}')
