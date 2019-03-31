import numpy as np
import pickle
import os
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Subset

def lbl_to_props(label):
    lbl_to_prop = dict()
    for l in label.unique():
        lbl_to_prop[l.item()] = ((label == l).sum().float() / label.nelement()).item()
    return lbl_to_prop

def filtering(label_or_props, min_num_lbl=2, min_prop_lbl=0.2):
    if type(label_or_props) is not dict:
        label_or_props = lbl_to_props(label_or_props)
    has_enough_num_lbl = len(label_or_props) >= min_num_lbl
    one_or_zero = lambda x: 1 if x > min_prop_lbl else 0 
    has_enough_prop_lbl = sum([one_or_zero(prop) for prop in label_or_props.values()]) >= min_num_lbl
    
    return \
        has_enough_num_lbl and \
        has_enough_prop_lbl

def to_subset_indices(dataset, prop_file_path, min_num_lbl=2, min_prop_lbl=0.2, force_reload=False):
    if not force_reload and os.path.exists(prop_file_path):
        dataset_proportions = pickle.load(open(prop_file_path, 'rb'))
    else:
        dataset_proportions = [lbl_to_props(data['label']) for data in tqdm(dataset)]
        pickle.dump(dataset_proportions, open(prop_file_path, 'wb'))
    
    f = lambda x: filtering(x, min_num_lbl=min_num_lbl, min_prop_lbl=min_prop_lbl)
    subset_mask = [f(prop) for prop in dataset_proportions]
    subset_indices = np.array(range(len(dataset)))[subset_mask]
    return subset_indices

def subset(dataset, prop_file_path, min_num_lbl=2, min_prop_lbl=0.2, force_reload=False):
    indices = to_subset_indices(
        dataset=dataset, 
        prop_file_path=prop_file_path,
        min_num_lbl=min_num_lbl, 
        min_prop_lbl=min_prop_lbl, 
        force_reload=force_reload)
    return Subset(dataset, indices)