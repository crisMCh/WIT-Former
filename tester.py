import torch
from tqdm import tqdm
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure
import wandb
from util.util import get_logger,mkdirs,save_images,make_dir,crop_center,ssim_xy,compute_ssim,compute_rmse
import pytorch_ssim
from util.image_utils import trunc, denormalize_, save_fig, batch_PSNR
from util.caculate_psnr_ssim import compute_measure

import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F



def mean_value(y_pred,y):
    y_pred=y_pred*3000-1000
    y_pred=torch.clip(y_pred,-160,240)
    y=y*3000-1000
    y=torch.clip(y,-160,240)
    return np.mean(abs(y-y_pred).detach().cpu().numpy())

def test(opt,model,loss_fn,testloader,device):
    import ipdb
    # ipdb.set_trace()
    if opt.mirror_padding is not None:
        opt.phase=opt.phase

    ###################################### Cristina's params ######################################
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    ori_psnr_list, ori_ssim_list, ori_rmse_list = [], [], []
    pred_psnr_list, pred_ssim_list, pred_rmse_list = [], [], []
    
    # Create directory for saving predictions
    predictions_dir = os.path.join(opt.save_img_dir, 'predictions', str(opt.name)+'_e'+str(opt.checkpoint)+'_n'+str(opt.noise_level))
    pred_dir_path = os.path.join(predictions_dir, 'npy')
    print(pred_dir_path)
    os.makedirs(pred_dir_path, exist_ok=True)

    ###################################### Cristina's params ######################################

    os.makedirs('./results/test_results/'+opt.name, exist_ok=True)
    os.makedirs('./results/test_results/'+opt.name+'/'+opt.phase+'-npy', exist_ok=True)

    test_logger = get_logger('./results/test_results/'+opt.name+'.log')
    test_logger.info('start testing!')
    
    Lambda=opt.lambda_factor

    up=torch.nn.Upsample(scale_factor=tuple([2.5,1,1]),mode='trilinear')

    ssim_loss = pytorch_ssim.SSIM3D(window_size = 11)
    #eval     
    test_running_psnr = []
    test_running_ssim3d=[]
    test_running_loss = 0   
    test_running_ssim2d=[]
    test_running_rmse=[]
    test_running_mean=[]
    test_losses=[]
    
    test_running_denrmse=[]

    model.eval()
    iters=0
    length=len(testloader.dataset) if opt.num_test==0 else opt.num_test
    


    with torch.no_grad():
        psnr_dataset = []
        psnr_model_init = []
        for ii, test_val in enumerate(tqdm(testloader), 0):

            # if iters >= opt.num_test:  # only apply our model to opt.num_test images.
            #     break
            x = test_val[0]
            y = test_val[1]
            
            
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            b,c,d,h,w= y_pred.shape
            orig_pred=y_pred
            y_pred = F.adaptive_avg_pool3d(y_pred, (3, h, w))
            test_loss = loss_fn(y_pred, y)+Lambda*(1-compute_ssim(y_pred, y))
            #test_loss = loss_fn(y_pred, y)
            test_losses.append(test_loss.item()) 
            
            test_psnr=peak_signal_noise_ratio(y_pred,y)
            test_ssim3d=pytorch_ssim.ssim3D(y_pred,y)
            test_ssim2d=compute_ssim(y_pred, y)
            test_rmse=compute_rmse(y_pred, y)
            test_running_loss += test_loss.item()
            test_running_psnr.append(test_psnr.detach().cpu().numpy())
            test_running_ssim3d.append(test_ssim3d.detach().cpu().numpy())
            test_running_ssim2d.append(test_ssim2d.detach().cpu().numpy())
            test_running_rmse.append(test_rmse.detach().cpu().numpy())
            test_running_mean.append(mean_value(y_pred, y))
        
            
            ###################################### Cristina's params ######################################
            # denormalize, truncate
            H, W = x.shape[-2], x.shape[-1]  # input shape [1, 3, 512, 512]
            # we're interested only in middle slice shape now: [1, 512, 512]
            input_ = x[:, :, 1, :, :]
            target = y[:, :, 1, :, :]
            #restored = y_pred[:, :, 1, :, :] 
            restored = orig_pred[:, :, 3, :, :] # because of upsampling            
            restored_model = y_pred
            
            psnr_dataset.append(batch_PSNR(x, y, False).item())
            psnr_model_init.append(batch_PSNR(y_pred, y, False).item())

            test_denrmse=compute_rmse(denormalize_(opt, y_pred.to(device)), denormalize_(opt,y.to(device)))
            test_running_denrmse.append(test_denrmse.detach().cpu().numpy())
            
            x = trunc(opt, denormalize_(opt, input_.view(H, W).cpu().detach()))
            y = trunc(opt, denormalize_(opt, target.view(H, W).cpu().detach()))
            pred = trunc(opt, denormalize_(opt, restored.view(H, W).cpu().detach()))

            
            
            os.makedirs(pred_dir_path, exist_ok=True)
            pred_file_path = os.path.join(pred_dir_path, f'prediction_{iters:04d}.npy')
            np.save(pred_file_path, pred.numpy())

            data_range = opt.trunc_max - opt.trunc_min

            original_result, pred_result = compute_measure(x, y, pred, data_range)

            ori_psnr_avg += original_result[0]
            ori_ssim_avg += original_result[1]
            ori_rmse_avg += original_result[2]
            pred_psnr_avg += pred_result[0]
            pred_ssim_avg += pred_result[1]
            pred_rmse_avg += pred_result[2]

            # Append results to lists
            ori_psnr_list.append(original_result[0])
            ori_ssim_list.append(original_result[1])
            ori_rmse_list.append(original_result[2])
            pred_psnr_list.append(pred_result[0])
            pred_ssim_list.append(pred_result[1])
            pred_rmse_list.append(pred_result[2])
            if opt.save_images:
                save_fig(opt, x, y, pred, ii, original_result, pred_result, predictions_dir)
                # Save comparison figure
                       
                #comparison_fig, comparison_ax = plt.subplots(1, orig_pred.shape[2], figsize=(30, 10))
                #for i in range(orig_pred.shape[2]):
                #    comparison_ax[i].imshow(trunc(opt, denormalize_(opt, orig_pred[:, :, i, :, :].view(H, W).cpu().detach())), cmap=plt.cm.gray, vmin=opt.trunc_min, vmax=opt.trunc_max)
                #    comparison_ax[i].set_title(f'Slice {i}', fontsize=30)
        
                #comparison_fig_path = os.path.join(predictions_dir, "fig", f'comparison_{ii:04d}.png')
                #comparison_fig.savefig(comparison_fig_path)
                #plt.close(comparison_fig)
                #print(f"Fig {ii} exported")
                
            if iters % 1== 0:  # save images to an HTML file
                test_logger.info('processing (%04d)-th image... ' % (iters))
                test_logger.info('(test_loss: %.6f, test_psnr: %.4f, test_ssim3d: %.4f,test_ssim2d: %.4f,test_rmse: %.6f,test_denrmse: %.6f) ' % (test_loss, test_psnr, test_ssim3d,test_ssim2d,test_rmse,test_denrmse))

            iters+=1
            
        psnr_dataset = sum(psnr_dataset)/length
        psnr_model_init = sum(psnr_model_init)/length
        print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_model_init))



    average_test_loss = test_running_loss /length*opt.test_batch_size
    average_test_psnr= np.mean(test_running_psnr)
    average_test_ssim3d=np.mean(test_running_ssim3d)
    average_test_ssim2d=np.mean(test_running_ssim2d)
    average_test_rmse=np.mean(test_running_rmse)
    average_test_mean=np.mean(test_running_mean)
    
    average_test_denrmse=np.mean(test_running_denrmse)
        
    test_message='(average_test_loss: %.6f, average_test_psnr: %.4f, average_test_ssim3d: %.4f,average_test_ssim2d:%.4f,average_test_rmse:%.6f,,average_test_denrmse:%.6f,average_test_mean:%.4f) ' % (average_test_loss, average_test_psnr, average_test_ssim3d,average_test_ssim2d,average_test_rmse,average_test_denrmse,average_test_mean)
    std_message='( std_test_psnr: %.4f, std_test_ssim3d: %.4f,std_test_ssim2d:%.4f,std_test_rmse:%.6f,std_test_mean:%.4f) ' % (np.std(test_running_psnr), np.std(test_running_ssim3d),np.std(test_running_ssim2d),np.std(test_running_rmse),np.std(test_running_mean))
    test_logger.info(test_message)
    test_logger.info(std_message)
    test_logger.info('finish!')
    
    # Create a DataFrame to save the results
    sample_results  = {
        'Index': list(range(length)),
        'Original_PSNR': ori_psnr_list,
        'Original_SSIM': ori_ssim_list,
        'Original_RMSE': ori_rmse_list,
        'Prediction_PSNR': pred_psnr_list,
        'Prediction_SSIM': pred_ssim_list,
        'Prediction_RMSE': pred_rmse_list,
        'Prediction_SSIM2D': test_running_ssim2d
    }

    df_samples = pd.DataFrame(sample_results)

    # DataFrame for averages
    avg_results = {
        'Metric': ['PSNR', 'SSIM', 'RMSE', 'SSIM2D'],
        'Original_Avg': [ori_psnr_avg/length, ori_ssim_avg/length, ori_rmse_avg/length, 0],
        'Prediction_Avg': [pred_psnr_avg/length, pred_ssim_avg/length, pred_rmse_avg/length, average_test_ssim2d]
    }
    df_averages = pd.DataFrame(avg_results)

    # Save both DataFrames to separate CSV files or combine them if needed
    result_samples_path = os.path.join(predictions_dir, 'measurement_sample_results.csv')
    result_averages_path = os.path.join(predictions_dir, 'measurement_avg_results.csv')

    df_samples.to_csv(result_samples_path, index=False)
    df_averages.to_csv(result_averages_path, index=False)

    print(f"Sample results saved to {result_samples_path}")
    print(f"Average results saved to {result_averages_path}")

    # Save test losses
    loss_file_path = os.path.join(predictions_dir, 'test_losses.npy')
    np.save(loss_file_path, np.array(test_losses))
    print(f"Test losses saved to {loss_file_path}")


