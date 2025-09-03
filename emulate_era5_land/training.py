"""
Script for dispatching single tracktrain-based training runs at a time,
each based on the configuration dict below. Each run will create a new
ModelDir directory and populate it with model info, the configuration,
and intermittent models saved duing training.
"""
import numpy as np
import json
from pathlib import Path
import torch
from emulate_era5_land.generators import SparseTimegridSampleDataset
from emulate_era5_land.generators import stsd_worker_init_fn
from emulate_era5_land.models import LSTM_S2S

metric_options = {
        "mae":torch.nn.L1Loss,
        "mse":torch.nn.MSELoss,
        }

model_options = {
        "lstm-s2s":LSTM_S2S
        }

optimizer_options = {
        "adam":torch.optim.Adam,
        "adamw":torch.optim.AdamW,
        "radam":torch.optim.RAdam,
        "rmsprop":torch.optim.rmsprop,
        "sgd":torch.optim.SGD,
        }

schedule_options = {
        "constant":torch.optim.lr_scheduler.ConstantLR,
        "linear":torch.optim.lr_scheduler.LinearLR,
        "exponential":torch.optim.lr_scheduler.ExponentialLR,
        "polynomial":torch.optim.lr_scheduler.PolynomialLR,
        "reduceonplateu":torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cyclic":torch.optim.lr_scheduler.CyclicLR,
        "coswarmrestart":torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        }

def get_model(model_type:str, model_args:dict={}):
    assert model_type in model_options.keys()
    return model_options.get(model_type)(**model_args)

def get_optimizer(optimizer_type:str, optimizer_args:dict={}):
    assert optimizer_type in optimizer_options.keys()
    return optimizer_options.get(optimizer_type)(**optimizer_args)

def get_lr_schedule(schedule_type:str, schedule_args:dict={}):
    assert schedule_type in schedule_options.keys()
    return schedule_options.get(schedule_type)(**schedule_args)

class EarlyStopper:
    """
    Save a state for the number of epochs of validation loss decrease
    """
    def __init__(self, patience=1, min_delta=0):
        """
        :@param patience: Number of validation epochs to allow no decrease
            before signaling for early stopping
        :@param min_delta: Extra buffer above or below threshold for
            incrementing the counter. If positive, loss may be higher than
            the minimum up to min_delta amount without incrementing the
            counter, and vice-versa if negative.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_single(config:dict, sequences_dir:Path, model_parent_dir:Path,
        use_residual_norm=False):
    """
    Dispatch the training routine for a single model configuration...
    """
    ## declare datasets for training and validation
    ds_train = SparseTimegridSampleDataset(
            timegrids=config["data"]["train"]["timegrids"],

            ## feature configuration
            window_feats=config["feats"]["window_feats"],
            horizon_feats=config["feats"]["horizon_feats"],
            target_feats=config["feats"]["target_feats"],
            static_feats=config["feats"]["static_feats"],
            static_int_feats=config["feats"]["static_int_feats"],
            aux_dynamic_feats=config["feats"]["aux_dynamic_feats"],
            aux_static_ceats=config["feats"]["aux_static_feats"],
            derived_feats=config["feats"]["derived_feats"],
            static_embed_maps=config["feats"]["static_embed_maps"],
            window_size=config["feats"]["window_size"],
            horizon_size=config["feats"]["horizon_size"],
            norm_coeffs=config["feats"]["norm_coeffs"],

            ## training data specific configuration
            shuffle=config["data"]["train"]["shuffle"],
            sample_cutoff=config["data"]["train"]["sample_cutoff"],
            sample_across_files=config["data"]["train"]["sample_across_files"],
            sample_under_cutoff=config["data"]["train"]["sample_under_cutoff"],
            sample_separation=config["data"]["train"]["sample_separation"],
            random_offset=config["data"]["train"]["random_offset"],
            chunk_pool_count=config["data"]["train"]["chunk_pool_count"],
            buf_size_mb=config["data"]["train"]["buf_size_mb"],
            buf_slots=config["data"]["train"]["buf_slots"],
            buf_policy=config["data"]["train"]["buf_policy"],

            seed=config["seed"],
            )

    ds_val = SparseTimegridSampleDataset(
            timegrids=config["data"]["val"]["timegrids"],

            ## feature configuration
            window_feats=config["feats"]["window_feats"],
            horizon_feats=config["feats"]["horizon_feats"],
            target_feats=config["feats"]["target_feats"],
            static_feats=config["feats"]["static_feats"],
            static_int_feats=config["feats"]["static_int_feats"],
            aux_dynamic_feats=config["feats"]["aux_dynamic_feats"],
            aux_static_ceats=config["feats"]["aux_static_feats"],
            derived_feats=config["feats"]["derived_feats"],
            static_embed_maps=config["feats"]["static_embed_maps"],
            window_size=config["feats"]["window_size"],
            horizon_size=config["feats"]["horizon_size"],
            norm_coeffs=config["feats"]["norm_coeffs"],

            ## validation data specific configuration
            shuffle=config["data"]["val"]["shuffle"],
            sample_cutoff=config["data"]["val"]["sample_cutoff"],
            sample_across_files=config["data"]["val"]["sample_across_files"],
            sample_under_cutoff=config["data"]["val"]["sample_under_cutoff"],
            sample_separation=config["data"]["val"]["sample_separation"],
            random_offset=config["data"]["val"]["random_offset"],
            chunk_pool_count=config["data"]["val"]["chunk_pool_count"],
            buf_size_mb=config["data"]["val"]["buf_size_mb"],
            buf_slots=config["data"]["val"]["buf_slots"],
            buf_policy=config["data"]["val"]["buf_policy"],

            seed=config["seed"],
            )

    ## initialize data loaders for training and validation
    dl_train = torch.utils.data.DataLoader(
            dataset=ds_train,
            batch_size=config["data"]["train"]["batch_size"],
            num_workers=config["data"]["train"]["num_workers"],
            prefetch_factor=config["data"]["train"]["prefetch_factor"],
            worker_init_fn=stsd_worker_init_fn,
            )

    dl_val = torch.utils.data.DataLoader(
            dataset=ds_val,
            batch_size=config["data"]["val"]["batch_size"],
            num_workers=config["data"]["val"]["num_workers"],
            prefetch_factor=config["data"]["val"]["prefetch_factor"],
            worker_init_fn=stsd_worker_init_fn,
            )

    ## initialize all the metric functions, which should include the loss func
    metrics = {k:metric_options[v[0]](**v[1]) for k,v in config["metrics"]}

    ## initialize the model, providing default args that would be redundant
    model = get_model(
            model_type=config["model"]["type"],
            model_args={
                ## defaults for feature sizes to prevent repetition
                "window_feats":config["feats"]["window_feats"],
                "horizon_feats":config["feats"]["horizon_feats"],
                "target_feats":config["feats"]["target_feats"],
                "static_feats":config["feats"]["static_feats"],
                "static_int_feats":config["feats"]["static_feats"],
                "static_embed_maps":config["feats"]["static_embed_maps"],
                "norm_coeffs":config["feats"]["norm_coeffs"],

                ## user-defined other model parameters
                **config["model"]["args"],
                },
            )
    optimizer = get_optimizer(
            optimizer_type=config["setup"]["optimizer_type"],
            optimizer_args={
                "params":model.parameters(),
                "lr":config["setup"]["initial_lr"],
                **config["setup"]["optimizer_args"],
                }
            )
    schedule = get_lr_schedule(
            schedule_type=config["setup"]["schedule_type"],
            schedule_args={
                "optimizer":optimizer,
                **config["setup"]["schedule_args"],
                }
            )
    stopper = EarlyStopper(
            patience=config["setup"]["early_stop_patience"],
            min_delta=config["setup"].get("early_stop_delta", 0.),
            )

    ## run the training loop
    early_stop = False
    dl_train_iter = iter(dl_train)
    dl_val_iter = iter(dl_val)
    metric_values = {
            "train":{mk:[] for mk in metrics.keys()},
            "val":{mk:[] for mk in metrics.keys()}
            "train_epochs":[],
            "val_epochs":[],
            }
    assert config["setup"]["loss_metric"] in metric_values.keys()
    for epoch in range(config["setup"]["max_epochs"]):
        ## train on a series of batches for this epoch.
        epoch_metrics = {mk:[] for mk in metrics.keys()}
        for bix in range(config["setup"]["batches_per_epoch"]):
            try:
                xt,yt,at = next(dl_train_iter)
            except StopIteration:
                dl_train_iter = iter(dl_train)
                xt,yt,at = next(dl_train_iter)
            w,h,s,si = xt
            pt = model(w, h, s, si)
            for mk,metric in metrics.items():
                tmpm = metric(pt,yt)
                if mk==config["setup"]["loss_metric"]:
                    optimizer.zero_grad()
                    ## calculate gradients wrt parameter tensors
                    loss.backward()
                    ## use gradients to update parameter tensors
                    optimizer.step()
                epoch_metrics[mk].append(tmpm)
        ## for now, store every batch's metrics. maybe consider average later.
        for mk,epms in epoch_metrics.items():
            metric_values["train"][mk].append(epms)
        metric_values["train_epochs"].append(epoch)

        ## run validation every val_frequency epochs
        if epoch % config["setup"]["val_frequency"] == 0:
            val_metrics = {mk:[] for mk in metrics.keys()}
            with torch.no_grad():
                for bix in range(config["setup"]["batches_per_epoch"]):
                    try:
                        xv,yv,av = next(dl_val_iter)
                    except StopIteration:
                        dl_val_iter = iter(dl_val)
                        xv,yv,av = next(dl_val_iter)
                    wv,hv,sv,siv = xv
                    pv = model(wv, hv, sv, siv)

                    for mk,metric in metrics.items():
                        tmpm = metric(pv,yv)
                        val_metrics[mk].append(tmpm)

                ## update validation metrics for this epoch
                for mk,epms in val_metrics.keys():
                    metric_values["val"][mk].append(epms)
                metric_values["val_epochs"].append(epoch)

            ## check for early stopping given all batches in this epoch
            early_stop = stopper.early_stop(np.average(
                metric_values["val"][config["setup"]["loss_metric"]][-1]
                ))
        if early_stop:
            break

if __name__=="__main__":
    ## config template
    info_era5 = json.load(
            proj_root.joinpath("data/list_feats_era5.json").open("r"))
    config = {
        "feats":{
            "window_feats":[
                "pres","tmp","dwpt","apcp","alb",
                "dlwrf","dswrf","weasd","windmag",
                "vsm-07", "vsm-28", "vsm-100", "vsm-289",
                ],
            "horizon_feats":[
                "pres","tmp","dwpt","apcp","alb",
                "dlwrf","dswrf","weasd","windmag",
                ],
            "target_feats":[
                "diff vsm-07","diff vsm-28","diff vsm-100","diff vsm-289",
                ],
            "static_feats":["geopot","lakec","vc-high","vc-low",],
            "static_int_feats":["soilt","vt-high","vt-low"],
            "aux_dynamic_feats":["evp","lhtfl"],
            "aux_static_feats":["vidxs","hidxs"],
            "derived_feats":info_era5["derived-feats"],
            "static_embed_maps":{
                "soilt":[0, 1, 2, 3, 4, 5, 6, 7],
                "vt-low":[0,  1,  2,  7,  9, 10, 11, 13, 16, 17],
                "vt-high":[0,  3,  5,  6, 18, 19],
                },
            "window_size":24,
            "horizon_size":72,
            "norm_coeffs":info_era5["norm-coeffs"],
            },
        "data":{
            "train":{
                "shuffle":True,
                "sample_cutoff":.66,
                "sample_across_files":True,
                "sample_under_cutoff":True,
                "sample_separation":157,
                "random_offset":True,
                "chunk_pool_count":48,
                "buf_size_mb":4096,
                "buf_slots":48,
                "buf_policy":0,
                "batch_size":256,
                "num_workers":5,
                "prefetch_factor":4,
                },
            "val":{
                "shuffle":True,
                "sample_cutoff":.66,
                "sample_across_files":True,
                "sample_under_cutoff":False,
                "sample_separation":157,
                "random_offset":True,
                "chunk_pool_count":48,
                "buf_size_mb":4096,
                "buf_slots":48,
                "buf_policy":0,
                "batch_size":256,
                "num_workers":5,
                "prefetch_factor":4,
                },
            },
        "metrics":{},
        "model":{
            "type":"lstm-s2s",
            "args":{
                "static_int_encoding_size":6,
                "num_hidden_feats":32,
                "num_hidden_layers":4,
                "normalized_inputs":True,
                "normalized_outputs":False,
                "cycle_targets":[
                    "diff vsm-07","diff vsm-28","diff vsm-100","diff vsm-289",
                    ],
                "teacher_forcing":True,
                },
            },
        "setup":{
            "loss_metric":None,
            "optimizer_type":None,
            "optimizer_args":{},
            "schedule_type":None,
            "schedule_args":None,
            "initial_lr":None,
            "early_stop_patience":None,
            "early_stop_delta":None,
            "max_epochs":None,
            "batches_per_epoch":None,
            "val_frequency":None,
            },
        "seed":None,
        "name":None,
        "notes":None,
        }
