from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import logging

import numpy as np
import xarray as xr
import pandas as pd
import time
import dask
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader

from config import CHLConfig
from models import CNN,UNet
from functions import Dataset_XY,Dataset_X,train_network,train_val_dataset,chloro_prediction,test_metrics,plot_trend
        
cs = ConfigStore.instance()
cs.store(name="chl_config", node=CHLConfig)
log = logging.getLogger(__name__)
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: CHLConfig) -> None:
    start_time = time.time()
    log.info(OmegaConf.to_yaml(cfg))
    # LOAD INPUT VARIABLES AND SCALE
    ###############################
    if cfg.data.input == 'obs':
        ds_input = xr.open_dataset(cfg.path.data_input + 'input_ds_sst_mld_ws.nc')
        ds_input = ds_input.transpose('time', 'latitude', 'longitude')
        a = ds_input.mask.where(ds_input.mask==3)
        mask = np.isnan(a)
        mask = ~mask   
        ds_input = ds_input.assign(variables={"mask": (('latitude','longitude'), mask.data)}) 
        
    # Normalize input
    #x_train = ds_input.sel(time=slice(cfg.data.d1_train, cfg.data.d2_train)).load()
    
    x_train = ds_input.sel(time = slice(cfg.data.d1_train,cfg.data.d2_train)).load()
    x_train = x_train.fillna(0)
    Xm   = x_train.sel(time = slice(cfg.data.d1_train,cfg.data.d2_train)).mean(skipna = True)
    Xstd = x_train.sel(time = slice(cfg.data.d1_train,cfg.data.d2_train)).std(skipna = True)
    x_scaled = (x_train - Xm)/Xstd
    n_channel = len(cfg.data.var_input)
    X_scaled = np.zeros([x_train.time.size,
                         x_train.latitude.size,
                         x_train.longitude.size,
                         n_channel])
    i = 0
    for v in cfg.data.var_input:
        X_scaled[:,:,:,i] = x_scaled[v].data.astype('float32')
        i+=1

    # LOAD CHL OUTPUT AND SCALE
    ###########################
    if cfg.data.output == 'globcolour_cmems':
        ds_out = xr.open_dataset(cfg.path.data_output + "Same_grid_CHL_25km.nc")
        ds_out = ds_out.rename({'CHL': 'chloro'})
        ds_out = ds_out.assign(variables={"mask": (('latitude','longitude'), ds_input.mask.data)}) 
        ds_out = ds_out.where(ds_out.mask == 1)

        # Set monthly start date
        new_index = pd.date_range(
            start=str(ds_out.time[0].data)[0:7],
            end=str(ds_out.time[-1].data)[0:7],
            freq='MS'
        )
        ds_out = ds_out.assign_coords(time=new_index)
        y_train = ds_out.sel(time=slice(cfg.data.d1_train, cfg.data.d2_train)).astype('float32').load()

        # # === Check CHL-a == 0 before log transform ==============================================================
        # chl_data = y_train['chloro'].data
        # num_chl_zero = np.sum(chl_data == 0)
        # print(f"[INFO] Number of CHL-a values exactly equal to 0 before log: {num_chl_zero}")

        # chl_data[chl_data == 0] = np.nan  # Replace 0 with NaN
        # y_train = y_train.assign(chloro=(('time', 'lat', 'lon'), chl_data))
        # # === Check CHL-a == 0 before log transform END ==============================================================

        
    # Log-transform output
    #Normalise output, log-transform, save in Y_scaled
    y_train = np.log(y_train)
    # Replace -inf with -10 (for CanESM5 CMIP model)
    chloro_inf = y_train['chloro'].where(~np.isinf(y_train['chloro']), other=-10)
    y_train = y_train.assign(variables={"chloro": (('time','latitude','longitude'), chloro_inf.data)})

    Cm   = y_train['chloro'].mean(skipna = True)
    Cstd = y_train['chloro'].std(skipna = True)
    C_scaled = (y_train['chloro'] - Cm)/Cstd
    Y_scaled = C_scaled.data

    print("X_scaled shape:", X_scaled.shape)
    print("Y_scaled shape:", Y_scaled.shape)
    
    #CHOICE OF MODEL
    ################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.params.model == 'CNN':
        model = CNN(n_channel = n_channel,
                    rateDropout=cfg.params.rateDropout).to(device)
    
    if cfg.out.training:
        #DATALOADERS
        ############
        dataloaders = train_val_dataset(X_scaled, Y_scaled, time = x_train.time,
                                        val_split=cfg.params.val_split,
                                        batch_size = cfg.params.batch_size,
                                        log = cfg.path.log,
                                        random_state=cfg.params.random_state)
        #TRAINING
        #########
        criterion = torch.nn.MSELoss()
        train_network(model,dataloaders, criterion, 
                      n_epochs = cfg.params.n_epochs,
                      lr       = cfg.params.lr,
                      path_log = cfg.path.log)
        log.info("Training done.")

    #PREDICTION
    ###########
    if cfg.out.prediction:
        log.info("Computing chlorophyl predictions...")
        chloro_prediction(ds_input,model,
                          log = cfg.path.log,
                          d1_pred = cfg.data.d1_pred, d2_pred = cfg.data.d2_pred,
                          Xm=Xm, Xstd=Xstd ,Cm=Cm, Cstd=Cstd,
                          var_input = cfg.data.var_input)
        log.info("Prediction of test dataset done.")

    #TEST METRICS
    #############
    if cfg.out.metric:
        log.info("Computing Metrics...")
        test_metrics(ds_out,log = cfg.path.log,
                     d1_test = cfg.data.d1_test, d2_test = cfg.data.d2_test)
        log.info(pd.read_csv(cfg.path.log +'Metrics.csv'))

    if cfg.out.plot:
        plot_trend(ds_out,
                   output = cfg.data.output,
                   input_name = cfg.data.input,
                   log = cfg.path.log,
                   d1_pred = cfg.data.d1_pred, d2_pred = cfg.data.d2_pred,
                   d1_train = cfg.data.d1_train, d2_train = cfg.data.d2_train,
                   t = cfg.data.plot_date)

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    hours, remainder = divmod(elapsed_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    log.info(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
        
if __name__ == "__main__":
    main()


