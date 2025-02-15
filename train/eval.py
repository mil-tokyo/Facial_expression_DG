from torch import nn
import torch
from tqdm import tqdm


def eval_model(model, eval_data, device, epoch, filename):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    data_num = 0
    for inputs, labels in tqdm(eval_data, ncols=100):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            data_num += inputs.size(0)
    epoch_loss = running_loss / len(eval_data.dataset)
    epoch_acc = running_corrects / len(eval_data.dataset)
    log = 'Eval: Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc)
    # logger.add_scalar('val/loss', epoch_loss, epoch)
    # logger.add_scalar('val/acc', epoch_acc, epoch)

    print(log)
    with open(filename, 'a') as f: 
        f.write(log + '\n')
    return epoch_loss, epoch_acc


def eval_demo(model, inputs, device):
    model.eval()
    inputs = inputs.to(device)

    output = model(inputs)
    if isinstance(output, tuple):
        output = output[0]
    output = nn.functional.softmax(output, dim=1)

    return output.to('cpu').detach()

