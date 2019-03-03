import matplotlib.pyplot as plt
import labelme
from torchvision import utils


def show_labeled_img(image, label, label_names):
    """Show labeled image"""
    img = (image * 255).astype('uint8')
    lbl = label.astype('uint8')
    print(img.shape, lbl.shape)
    lbl_viz = labelme.utils.draw_label(lbl, img, label_names)
    plt.imshow(lbl_viz)

# Helper function to show a batch
def show_batch(sample_batched, label_names):
    """Show labeled image for a batch of samples."""
    images_batch, labels_batch = sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    n_row = 2

    img_grid = utils.make_grid(images_batch, nrow=n_row)
    img_grid = img_grid.numpy().transpose((1, 2, 0))
    
    lbl_grid = utils.make_grid(labels_batch.unsqueeze(1), nrow=n_row))
    lbl_grid = lbl_grid.numpy()[0]
    
    #lbl_names = sorted(list({l for group in sample_batched['label_names'] for l in group}))
    
    show_labeled_img(img_grid, lbl_grid, label_names)
    plt.title('Batch from dataloader')