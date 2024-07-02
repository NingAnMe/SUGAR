import nibabel as nib
import os
from utils.utils import abspath


def get_weight(feature, hemi):
    if feature == 'sulc':
        weight_data_path = os.path.join(abspath, 'auxi_data', f'{hemi}.904_weight.std.sulc')
    elif feature == 'curv':
        weight_data_path = os.path.join(abspath, 'auxi_data', f'{hemi}.904_weight.std.curv')

    weight = nib.freesurfer.read_morph_data(weight_data_path)
    return weight

