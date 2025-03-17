import torch
from tqdm import tqdm
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure
import wandb
from util.util import get_logger,mkdirs,compute_ssim,compute_rmse,compute_psnr2D, load_checkpoint, load_start_epoch
import os
import math
import pytorch_ssim
import ipdb
from glob import glob
import torch.nn.functional as F


def train(opt,model,optimizer,lr_scheduler,loss_fn,trainloader,testloader,device):

    if opt.resume: 
        model_root = opt.model_path
        model_path = glob(os.path.join(model_root, str(opt.checkpoint) + "*"))[0]

        if not model_path:
            raise ValueError(f"No model with checkpoint {opt.checkpoint} found in {model_root}")
        else:		 
            print(f"Resuming from {model_path}")

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cuda:{}'.format(opt.gpu_ids[0]))

            # Load model weights (handles single & multi-GPU)
            if len(opt.gpu_ids) > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # Load optimizer (prevents sudden loss increase)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler (ensures LR schedule continuity)
            if 'scheduler_state_dict' in checkpoint and lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Resume from the last saved epoch
            start_epoch = checkpoint.get('epoch', opt.checkpoint) + 1
            print(f"Training will resume from epoch {start_epoch}")

    else: 
        print("Training from scratch")
        start_epoch = 0

    
    if opt.use_wandb:
        wandb.init(project='litformer_review',name=opt.name)
        wandb.watch(model)
    train_logger = get_logger(opt.checkpoints_dir+'/'+opt.name+'/train.log')
    save_images_root='./results/'+opt.name
    mkdirs(save_images_root)  
    train_logger.info(model)
    train_logger.info('start training!')

    train_total_iters = 0
    val_total_iters = 0
    Lambda=opt.lambda_factor
    for epoch in tqdm(range(start_epoch, opt.epochs + 1)):
        train_length=len(trainloader.dataset)
        val_length=len(testloader.dataset)
        epoch_iter = 0                  #the number of training iterations in current epoch, reset to 0 every epoch
        
        running_loss = 0
        running_psnr3d=0
        running_pnsr2d=0
        running_ssim3d=0
        running_ssim2d=0
        

        model.train()

        if epoch>1:
            for x, y in tqdm(trainloader):

                x, y = x.to(device), y.to(device)

                B, C, D, H, W = x.shape
                y_pred = model(x)
                y_pred = F.adaptive_avg_pool3d(y_pred, (3, H, W))
                
                train_total_iters += 1
                epoch_iter += 1
                
                #ch_loss = loss_fn(y_pred, y)
                #ssim_loss = Lambda*(1-compute_ssim(y_pred, y))
                #train_loss = ch_loss
                train_loss = loss_fn(y_pred, y)+Lambda*(1-compute_ssim(y_pred, y))
                train_psnr3d=peak_signal_noise_ratio(y_pred,y)
                # train_psnr2d=compute_psnr2D(y_pred, y)
                train_ssim3d=pytorch_ssim.ssim3D(y_pred,y)
                train_ssim2d=compute_ssim(y_pred, y)
                

                optimizer.zero_grad()
                train_loss.backward()
                
                
                optimizer.step()

                #lr_scheduler.step()
                with torch.no_grad():
                    running_loss += train_loss.item()
                    running_psnr3d += train_psnr3d
                    running_ssim2d+=train_ssim2d
                    running_ssim3d+=train_ssim3d
                    

                if train_total_iters % opt.print_freq == 0:
                    message='(epoch: %d, iters: %d,epoch_loss: %.4f, train_psnr3d: %.4f,train_ssim3d: %.4f,train_ssim2d:.%.4f) ' % (epoch, epoch_iter,train_loss, train_psnr3d,train_ssim3d,train_ssim2d)
                    print(message)
                    if opt.use_wandb:
                        wandb.log({"train_loss": train_loss,
                                    'train_psnr':train_psnr3d,
                                    'train_ssim':train_ssim3d} )
                    
            epoch_loss = running_loss/train_length*opt.train_batch_size
            epoch_psnr3d= running_psnr3d/train_length*opt.train_batch_size
            epoch_ssim3d=running_ssim3d/train_length*opt.train_batch_size
            epoch_ssim2d=running_ssim2d/train_length*opt.train_batch_size
            

            #train_message='(epoch: %d, iters: %d,epoch_loss: %.4f, train_psnr: %.4f, train_ssim: %.4f) ' % (epoch, epoch_iter,epoch_loss, epoch_psnr, epoch_ssim)
            train_logger.info('Epoch: [{}/{}],epoch_loss: {:.6f}, train_psnr3d: {:.4f},train_ssim3d: {:.4f},train_ssim2d:{:.4f}'.format(epoch ,opt.epochs, epoch_loss, epoch_psnr3d, epoch_ssim3d, epoch_ssim2d))

        #eval  
        # if epoch>30:
        print('validation:')   
        test_running_psnr3d = 0
        test_running_ssim3d=0
        test_running_loss = 0 
        test_running_ssim2d=0
        test_running_rmse=0

        #ipdb.set_trace()
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(testloader):
                val_total_iters+=1
                x, y = x.to(device), y.to(device)
                B, C, D, H, W = x.shape
                y_pred = model(x)
                y_pred = F.adaptive_avg_pool3d(y_pred, (3, H, W))
                test_loss=loss_fn(y_pred, y)+Lambda*(1-compute_ssim(y_pred, y))
                test_psnr3d=peak_signal_noise_ratio(y_pred,y)
                test_ssim3d=pytorch_ssim.ssim3D(y_pred,y)
                test_ssim2d=compute_ssim(y_pred, y)
                test_rmse=compute_rmse(y_pred, y)
                
                test_running_loss += test_loss.item()
                test_running_psnr3d+=test_psnr3d
                test_running_ssim3d+=test_ssim3d
                test_running_ssim2d+=test_ssim2d
                test_running_rmse+=test_rmse
                    

                if val_total_iters % (opt.print_freq/2) == 0:
                    #print('(test_loss: %.6f, test_psnr: %.3f, test_ssim: %.5f) ' % (test_loss, test_psnr, test_ssim))
                    if opt.use_wandb:
                        wandb.log({"test_loss": test_loss,
                                    'test_psnr':test_psnr3d,
                                    'test_ssim':test_ssim3d} ) 
        epoch_test_loss = test_running_loss /val_length
        epoch_test_psnr3d= test_running_psnr3d/val_length
        epoch_test_ssim3d=test_running_ssim3d/val_length
        epoch_test_ssim2d=test_running_ssim2d/val_length 
        epoch_test_rmse=test_running_rmse/val_length      

        train_logger.info('val:Epoch: [{}/{}],epoch_loss: {:.6f}, val_psnr3d: {:.4f}, val_ssim3d: {:.4f},val_ssim2d: {:.4f},test_rmse: {:.6f}'.format(epoch , opt.epochs, epoch_test_loss, epoch_test_psnr3d,epoch_test_ssim3d,epoch_test_ssim2d,epoch_test_rmse))

        
        lr_scheduler.step()
        if opt.use_wandb:
            wandb.log({"epoch_train_loss": epoch_loss,
                        'epoch_train_psnr':epoch_psnr3d,
                        'epoch_train_ssim':epoch_ssim3d,
                        "epoch_test_loss":epoch_test_loss,
                        'epoch_test_psnr':epoch_test_psnr3d,
                        'epoch_test_ssim':epoch_test_ssim3d,
                        'epoch':epoch
                        } )
#save model
        if epoch % 5 == 0 and epoch > 0:
            if len(opt.gpu_ids) > 1:
                static_dict = model.module.state_dict()  # Multi-GPU
            else:
                static_dict = model.state_dict()  # Single-GPU

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': static_dict,  # Save model weights
                'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer
                'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,  # Save scheduler (if used)
                'train_loss': epoch_loss,
                'train_psnr': epoch_psnr3d,
                'train_ssim': epoch_ssim3d
            }

            save_path = f"{opt.checkpoints_dir}/{opt.name}/{epoch}_trainloss_{epoch_loss:.6f}_train_psnr{epoch_psnr3d:.3f}_train_ssim.pth"
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved: {save_path}")

    train_logger.info('finish training!')



