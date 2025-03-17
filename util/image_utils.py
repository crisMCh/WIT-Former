import torch
import numpy as np
import pickle
#import cv2
import matplotlib.pyplot as plt
import os


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

#def load_img(filepath):
#    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
#    img = img.astype(np.float32)
#    img = img/255.
#    return img

#def save_img(filepath, img):
#    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def denormalize_(opt, image):
        image = image * (opt.norm_range_max - opt.norm_range_min) + opt.norm_range_min
        return image



def save_fig(self, x, y, pred, fig_name, original_result, pred_result, save_path):
		x, y, pred = x.numpy(), y.numpy(), pred.numpy()
		f, ax = plt.subplots(1, 3, figsize=(30, 10))
		ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
		ax[0].set_title(f'Low dose {self.noise_level}', fontsize=30)
		ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
																		   original_result[1],
																		   original_result[2]), fontsize=20)
		ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
		ax[1].set_title('Result', fontsize=30)
		ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
																		   pred_result[1],
																		   pred_result[2]), fontsize=20)
		ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
		ax[2].set_title('Full-dose', fontsize=30)
		save_fig_path = os.path.join(save_path, 'fig')
		os.makedirs(save_fig_path, exist_ok=True)
		f.savefig(os.path.join(save_fig_path, 'result_{}.png'.format(fig_name)))
		plt.close()

def trunc(opt, mat):
		mat[mat <= opt.trunc_min] = opt.trunc_min
		mat[mat >= opt.trunc_max] = opt.trunc_max
		return mat