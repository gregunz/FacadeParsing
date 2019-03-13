import matplotlib.pyplot as plt
import labelme
import torch
from torchvision import utils

def convert_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy().transpose((1, 2, 0))
    return tensor

def show_img(image):
    if isinstance(image, torch.Tensor):
        image = image.clone().permute(1,2,0)
    plt.imshow(image.squeeze())
    plt.show()

def show_labeled_img(image, label, label_names='0123456789'):
    """Show labeled image"""  
    if isinstance(label_names, dict):
        label_names = {v:k for k, v in label_names.items()}
        label_names = [label_names[i] for i in label_names]

    image = convert_to_numpy(image)
    label = convert_to_numpy(label)
    img = (image * 255).astype('uint8')
    lbl = label.astype('uint8')
    print(img.shape, lbl.shape)
    if len(lbl.shape) == 3:
        lbl = lbl[:, :, 0]
    lbl_viz = labelme.utils.draw_label(lbl, img, label_names)
    plt.imshow(lbl_viz)
    
# Helper function to show a batch
def show_batch(sample_batched, label_names, nrow=2):
    """Show labeled image for a batch of samples."""
    images_batch, labels_batch = sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    img_grid = utils.make_grid(images_batch, nrow=nrow)
    img_grid = img_grid.numpy().transpose((1, 2, 0))
    
    lbl_grid = utils.make_grid(labels_batch, nrow=nrow)
    lbl_grid = lbl_grid.numpy()[0]
    
    #lbl_names = sorted(list({l for group in sample_batched['label_names'] for l in group}))
    
    show_labeled_img(img_grid, lbl_grid, label_names)
    plt.title('Batch from dataloader')