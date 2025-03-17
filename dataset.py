#dataset
import os, glob, shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from util import transforms
from util.image_utils import denormalize_
from util.caculate_psnr_ssim import compute_SSIM
import ipdb
import random
random.seed(42)

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list

    return tmplist

def random_sample(input_list, sample_size):
    if sample_size > len(input_list):
        sample_size = len(input_list)
    return random.sample(input_list, sample_size)


class Mayo_Dataset(Dataset):
    def __init__(self, opt, transforms=None):
        #ipdb.set_trace()
        self.transforms = transforms
        #hu_min, hu_max = hu_range
        self.phase=opt.phase
        self.mirror_padding=opt.mirror_padding

        self.q_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/quarter_lr/*')
        self.f_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/full_hr/*')


    def __getitem__(self, index):
        f_data=np.load(self.f_path_list[index]).astype(np.float32)
        q_data = np.load(self.q_path_list[index]).astype(np.float32)

        if self.transforms is not None:
            f_data = self.transforms[1](f_data)
            q_data = self.transforms[0](q_data)
        return q_data, f_data

    def __len__(self):
        return len(self.q_path_list)
    
    
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, noise_level, img_options=None, transforms=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = transforms
        self.opt=img_options
        
        self.noise_level=noise_level
        
        gt_dir = 'groundtruth' 
        input_dir = os.path.join('input', str(self.noise_level))  # Use noise level as subdirectory

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_numpy_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)  for x in noisy_files if is_numpy_file(x)]
        
        self.clean_dir = os.path.join(rgb_dir, gt_dir)
        self.noisy_dir = os.path.join(rgb_dir, input_dir)

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        
        if self.opt.multi_slice:
            clean_current = torch.from_numpy(np.float32(load_npy(self.clean_filenames[tar_index]))).unsqueeze(0)  # Shape: (1, H, W)
            noisy_current = torch.from_numpy(np.float32(load_npy(self.noisy_filenames[tar_index]))).unsqueeze(0)  # Shape: (1, H, W)
        else:
            clean_current = torch.from_numpy(np.float32(load_npy(self.clean_filenames[tar_index]))) 
            noisy_current = torch.from_numpy(np.float32(load_npy(self.noisy_filenames[tar_index])))
        
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

            
        
        
        # Load the previous, current, and next files for both clean and noisy
        if self.opt.multi_slice:
            next_file_name_clean, prev_file_name_clean = get_next_and_prev_filenames(clean_filename, self.clean_dir,self.opt)
            if self.opt.isTrain == False:
                next_file_name_noisy = next_file_name_clean
                prev_file_name_noisy = prev_file_name_clean
            else:
                next_file_name_noisy = next_file_name_clean.replace("_target.npy", "_input.npy")
                prev_file_name_noisy = prev_file_name_clean.replace("_target.npy", "_input.npy")
                
            clean_prev = torch.from_numpy(np.float32(load_npy(os.path.join(self.clean_dir, prev_file_name_clean)))).unsqueeze(0)  # Shape: (1, H, W)
            clean_next = torch.from_numpy(np.float32(load_npy(os.path.join(self.clean_dir, next_file_name_clean)))).unsqueeze(0)  # Shape: (1, H, W)
            noisy_prev = torch.from_numpy(np.float32(load_npy(os.path.join(self.noisy_dir, prev_file_name_noisy)))).unsqueeze(0)  # Shape: (1, H, W)
            noisy_next = torch.from_numpy(np.float32(load_npy(os.path.join(self.noisy_dir, next_file_name_noisy)))).unsqueeze(0)  # Shape: (1, H, W)
            # Create clean volume (previous, current, next)
            clean = torch.stack([clean_prev, clean_current, clean_next], dim=0).permute(1, 0, 2, 3)  # Shape: (1, 3, H, W)
            #clean = torch.stack([clean_current], dim=0)  # Shape: (1, 1, H, W)
    
            # Create noisy volume (previous, current, next)
            noisy = torch.stack([noisy_prev, noisy_current, noisy_next], dim=0).permute(1, 0, 2, 3)  # Shape: (1, 3, H, W)
            #noisy = torch.stack([noisy_current], dim=0)  # Shape: (1, 1, H, W)
        else:
            
            clean = clean_current
            noisy = noisy_current
            clean = torch.stack([clean_current, clean_current, clean_current], dim=0).unsqueeze(0)  # Shape: (1, 3, H, W)
            noisy = torch.stack([noisy_current, noisy_current, noisy_current], dim=0).unsqueeze(0)  # Shape: (1, 3, H, W)

        
        return noisy, clean
    
    
    
def get_next_and_prev_filenames(file_name, file_directory, opt, check_sim=True):
    """
    Get the next and previous filenames for the given file, ensuring they are similar based on SSIM.
    
    Parameters:
    ----------
    file_name : str
        The current filename.
    file_directory : str
        The directory where the files are located.
    opt : dict
        Options for similarity computation.

    Returns:
    -------
    tuple
        The next and previous filenames.
    """
    import os
    import re

    # Extract the base name, number, and file extension
    match = re.search(r"_I(\d+)_", file_name)
    if not match:
        raise ValueError("Filename does not contain a valid '_I<number>_' structure.")
    
    number_str = match.group(1)
    number = int(number_str)
    prefix, suffix = file_name.split(f"_I{number_str}_", 1)
    suffix = suffix.rstrip('.npy')  # Remove .npy for consistency
    file_extension = ".npy"  # Save the file extension for later use
    
    # Compute next and previous numbers
    next_number = number + 1
    prev_number = max(0, number - 1)

    # Format numbers with leading zeros
    next_number_str = f"{next_number:0{len(number_str)}d}"
    prev_number_str = f"{prev_number:0{len(number_str)}d}"

    # Construct next and previous filenames
    next_file_name = f"{prefix}_I{next_number_str}_{suffix}{file_extension}"
    prev_file_name = f"{prefix}_I{prev_number_str}_{suffix}{file_extension}"

     # Check if files exist
    next_file_path = os.path.join(file_directory, next_file_name)
    prev_file_path = os.path.join(file_directory, prev_file_name)
    

    if not os.path.exists(next_file_path):
        next_file_name = file_name  # Set next to current if file doesn't exist
    elif check_sim and not similar(file_name, next_file_name, file_directory, opt):
        next_file_name = file_name  # Set next to current if not similar

    if not os.path.exists(prev_file_path):
        prev_file_name = file_name  # Set previous to current if file doesn't exist
    elif check_sim and not similar(file_name, prev_file_name, file_directory, opt):
        prev_file_name = file_name  # Set previous to current if not similar# Verify similarity
    if not similar(file_name, next_file_name, file_directory, opt) and check_sim:
        next_file_name = file_name  # Set next to current if not similar
    if not similar(file_name, prev_file_name, file_directory, opt) and check_sim:
        prev_file_name = file_name  # Set previous to current if not similar

    return next_file_name, prev_file_name  

def similar(file_name1, file_name2, file_directory, opt):
    file1 = torch.tensor(denormalize_(opt, np.load(os.path.join(file_directory, file_name1))),dtype=torch.float32)
    file2 =  torch.tensor(denormalize_(opt, np.load(os.path.join(file_directory, file_name2))), dtype=torch.float32)
    ssim_val = compute_SSIM(file1, file2, opt.norm_range_max-opt.norm_range_min)
    return ssim_val > 0.5
    
    

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def load_npy(filepath):
    img = np.load(filepath)
    return img