import time
import sys
import copy
import torch

from loss_and_metric import jaccard_index

from tqdm import tqdm_notebook as tqdm

def train_model(dataloaders, dataset_sizes, path_to_data, model_name, model, device, criterion, optimizer, scheduler, writer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_jacc_mean = 0.0
    best_loss = sys.maxsize

    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_jacc = torch.tensor([0.0 for i in range(model.num_classes)])

                # Iterate over data.
                for i, sample in enumerate(tqdm(dataloaders[phase])):
                    inputs = sample['image'].to(device)
                    labels = sample['label'].squeeze().to(device)
                    #print(inputs.shape, labels.shape)
                    #print(set(labels.numpy().flatten()))

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    
                    # statistics
                    loss_scalar =  loss.item() * inputs.size(0)
                    running_loss += loss_scalar
                    running_jacc += jaccard_index(preds, labels, model.num_classes)

                    # tensorboard
                    x_axis = i + epoch * dataset_sizes[phase]
                    writer.add_scalar('{}_loss'.format(phase), loss_scalar,  x_axis)
                    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_jacc = running_jacc / dataset_sizes[phase]
                epoch_jacc_mean = epoch_jacc.mean().item()
                
                jacc_dict = {j:t.item() for j,t in enumerate(epoch_jacc)}
                print('{} Loss: {:.4f} MeanJacc: {:.4f}  JaccPerClass: {}'.format(phase, epoch_loss, epoch_jacc_mean, jacc_dict))

                # deep copy the model
                if phase == 'val' and (epoch_loss < best_loss or best_jacc_mean > epoch_jacc_mean):
                    best_jacc_mean = max(epoch_jacc_mean, best_jacc_mean)
                    best_loss = min(epoch_loss, best_loss)
                    model_path = '{}/models/model_{}_{}.torch'.format(path_to_data, model_name, epoch)
                    torch.save(model.state_dict(), model_path)
                    print('Saving model in {}'.format(model_path))
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

    except KeyboardInterrupt:
        pass
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_jacc_mean))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model