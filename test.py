from dataset import Mayo_Dataset, DataLoaderTrain
from torch.utils.data import Dataset, DataLoader
from util import transforms
from util.util import create_optimizer,CharbonnierLoss
import torch
from tester import test
import wandb
import time
from options.test_options import TestOptions
import numpy as np
import os
from glob import glob
from witformer import WITFormer


if __name__ == '__main__':

    min_value=-1000
    max_value=1000
    
    val_raw_transformer=transforms.Compose([
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    val_label_transformer=transforms.Compose([
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    val_transforms=[val_raw_transformer,val_label_transformer]

    opt = TestOptions().parse() 
    device=torch.device('cuda:{}'.format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    #device=torch.device("cpu") 

    test_dataset=DataLoaderTrain(opt.test_dir,opt.noise_level,opt,transforms=val_transforms)
    test_dataloader=DataLoader(test_dataset,batch_size=opt.test_batch_size,shuffle=False,num_workers=opt.num_threads)

    test_model=WITFormer(in_channels=1,out_channels=1,n_channels=64,num_heads_s=[1,2,4,8],num_heads_t=[1,2,4,8],res=True,attention_s=True,attention_t=True).to(device)

    #if len(opt.gpu_ids)>1:
    #    test_model=torch.nn.DataParallel(test_model,device_ids=opt.gpu_ids)

    model_root=opt.model_path
    model_path = glob(os.path.join(model_root, str(opt.checkpoint) + "*"))[0]
    
    if model_path is None:
        raise ValueError(f"No model with checkpoint {opt.checkpoint} found in {model_root}")
    else:
        print(f"Restoring at checkpoint {opt.checkpoint} from {model_path}")
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(opt.gpu_ids[0]))
        if len(opt.gpu_ids)>1:
            test_model=torch.nn.DataParallel(test_model,device_ids=opt.gpu_ids)
            test_model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            test_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        


    
    #if opt.rep:
        #test_model=repvgg_model_convert(test_model)
        
    loss_fn= CharbonnierLoss()
    #loss_fn=MSELoss
    test(
        opt, 
        model=test_model,
        loss_fn=loss_fn,
        testloader=test_dataloader,
        device=device,
        )

