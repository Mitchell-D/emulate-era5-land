"""
Script for dispatching single tracktrain-based training runs at a time,
each based on the configuration dict below. Each run will create a new
ModelDir directory and populate it with model info, the configuration,
and intermittent models saved duing training.
"""
import pickle as pkl
import numpy as np
import json
from pathlib import Path
import torch
from emulate_era5_land.generators import SparseTimegridSampleDataset
from emulate_era5_land.generators import stsd_worker_init_fn
from emulate_era5_land.generators import get_datasets_from_config
from emulate_era5_land.models import AccLSTM,get_model_from_config
from emulate_era5_land.evaluators import EvalSampleSources
from emulate_era5_land.helpers import move_to_device

def get_optimizer(optimizer_type:str, optimizer_args:dict={}):
    assert optimizer_type in optimizer_options.keys(),optimizer_options.keys()
    return optimizer_options.get(optimizer_type)(**optimizer_args)

def get_lr_schedule(schedule_type:str, schedule_args:dict={}):
    assert schedule_type in schedule_options.keys(),schedule_options.keys()
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

class FeatureWeightedL1Loss(torch.nn.Module):
    """
    Simple class for applying independent coefficient weights to
    """
    def __init__(self, feature_weights):
        super(FeatureWeightedL1Loss, self).__init__()
        self.register_buffer("w",torch.Tensor(feature_weights))

    def forward(self, prediction, target):
        return torch.mean(torch.abs(prediction - target) * self.w)

metric_options = {
        "mae":torch.nn.L1Loss,
        "mse":torch.nn.MSELoss,
        "fwmae":FeatureWeightedL1Loss,
        }

optimizer_options = {
        "adam":torch.optim.Adam,
        "adamw":torch.optim.AdamW,
        "radam":torch.optim.RAdam,
        "nadam":torch.optim.NAdam,
        "rmsprop":torch.optim.RMSprop,
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


def train_single(config:dict, model_parent_dir:Path, device=None, debug=False):
    """
    Dispatch the training routine for a single model configuration...
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## declare datasets for training and validation
    datasets = get_datasets_from_config(config)

    ## initialize data loaders for training and validation
    dl_train = torch.utils.data.DataLoader(
            dataset=datasets["train"],
            batch_size=config["data"]["train"]["batch_size"],
            num_workers=config["data"]["train"]["num_workers"],
            prefetch_factor=config["data"]["train"]["prefetch_factor"],
            worker_init_fn=stsd_worker_init_fn,
            )

    dl_val = torch.utils.data.DataLoader(
            dataset=datasets["val"],
            batch_size=config["data"]["val"]["batch_size"],
            num_workers=config["data"]["val"]["num_workers"],
            prefetch_factor=config["data"]["val"]["prefetch_factor"],
            worker_init_fn=stsd_worker_init_fn,
            )

    ## initialize all the metric functions, which should include the loss func
    metrics = {k:metric_options[k](**v).to(device)
            for k,v in config["metrics"].items()}

    ## initialize the model, providing default args that would be redundant
    model = get_model_from_config(config)

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

    ## Declare a EvalSampleSources instance to keep track of t/v data sources
    params = {
            "vidx_feat":("aux-static", "vidxs"),
            "hidx_feat":("aux-static", "hidxs"),
            "time_feat":("time", "epoch"),
            "cov_feats":[], "cov_reduce_metric":None, "cov_reduce_axes":(1,),
            }
    dataset_feats = {
            "aux-static":config["feats"]["aux_static_feats"], "time":["epoch"],
            }
    ess_t_name = (config["name"], "sample-sources", "train")
    ess_t = EvalSampleSources(
            params=params,
            feats=dataset_feats,
            meta={"model_config":config, "name":"_".join(ess_t_name)},
            )
    ess_v_name = (config["name"], "sample-sources", "val")
    ess_v = EvalSampleSources(
            params=params,
            feats=dataset_feats,
            meta={"model_config":config, "name":"_".join(ess_v_name)},
            )

    model = model.to(device)

    ## initialize the model directory, config json, and metric storage json.
    model_dir = model_parent_dir.joinpath(config["name"])
    assert not model_dir.exists(), f"Model directory exists: {model_dir.name}"
    model_dir.mkdir()
    config_path = model_dir.joinpath(f"{config['name']}_config.json")
    json.dump(config, config_path.open("w"), indent=2)
    metric_json_path = model_dir.joinpath(
            f"{config['name']}_metrics_simple.json")
    ## update the batch-wise metrics pkl
    metric_pkl_path = model_dir.joinpath(f"{config['name']}_metrics_all.pkl")

    ## run the training loop
    early_stop = False
    dl_train_iter = iter(dl_train)
    dl_val_iter = iter(dl_val)

    ## epoch-wise metrics for the 'simple' json, updated once per epoch
    metrics_epoch = {
            "train":{mk:[] for mk in metrics.keys()},
            "val":{mk:[] for mk in metrics.keys()},
            "train_epochs":[],
            "val_epochs":[],
            "lr":[],
            }
    min_loss = float("inf")
    lm = config["setup"]["loss_metric"]
    assert lm in metrics_epoch["train"].keys(),\
            f"{lm = } must match the key of a defined metric."

    ## batch-wise metrics for the pkl, updated once per epoch
    metrics_batch = {
            "train":{mk:[] for mk in metrics.keys()},
            "val":{mk:[] for mk in metrics.keys()},
            "lr":[],
            }
    for epoch in range(config["setup"]["max_epochs"]):
        if debug:
            print(f"Starting Epoch {epoch}")

        ## make a new list for batch-wise metric entries within this epoch
        for mk in metrics_batch["train"].keys():
            metrics_batch["train"][mk].append([])

        ## train on a series of batches for this epoch.
        model.train()
        for bix in range(config["setup"]["batches_per_epoch"]):
            ## draw a batch from the training data loader
            try:
                xt,yt,at = move_to_device(next(dl_train_iter), device)
            except StopIteration:
                if debug:
                    print(f"Re-initializing training generator!")
                dl_train_iter = iter(dl_train)
                xt,yt,at = move_to_device(next(dl_train_iter), device)
            wt,ht,st,sit,initt = xt
            yt, = yt

            ## run the model
            pt = model(wt, ht, st, sit, yt, device=device)

            ## evaluate the metrics
            for mk,metric in metrics.items():
                tmpm = metric(pt,yt)
                if mk==lm:
                    optimizer.zero_grad()
                    ## calculate gradients wrt parameter tensors
                    tmpm.backward()
                    ## use gradients to update parameter tensors
                    optimizer.step()
                ## update dicts tracking training progress
                tmpm_np = tmpm.cpu().detach().numpy()
                metrics_batch["train"][mk][-1].append(tmpm_np)

            ## update the sample tracking evaluator
            a_d,a_s,t = at
            ess_t.add_batch({
                "aux-static":a_s.cpu().detach().numpy(),
                "time":t.cpu().detach().numpy(),
                })

        ## Aggregate batch-wise metrics for simpler epoch-wise stats
        for mk in metrics_batch["train"].keys():
            metrics_epoch["train"][mk].append([
                float(np.average(metrics_batch["train"][mk][-1])),
                float(np.std(metrics_batch["train"][mk][-1])),
                ])

        ## update epoch-wise epoch counter and learning rate
        metrics_epoch["train_epochs"].append(epoch)
        metrics_epoch["lr"].append(schedule.get_last_lr())
        metrics_batch["lr"].append(schedule.get_last_lr())
        schedule.step()

        ## save the model if training loss went down. validation is more
        ## traditional, but I can manually select the model at the bottom of
        ## the valley of generality
        if min_loss > metrics_epoch["train"][lm][-1][0]:
            torch.save(
                model.state_dict(),
                model_dir.joinpath(f"{config['name']}_state_{epoch:04}.pwf")
                )

        ## run validation every val_frequency epochs
        if epoch % config["setup"]["val_frequency"] == 0:
            ## make a new list for batch-wise metric entries within this epoch
            for mk in metrics_batch["val"].keys():
                metrics_batch["val"][mk].append([])

            ## disable gradient tracking and execute validation
            model.eval()
            with torch.no_grad():
                for bix in range(config["setup"]["batches_per_epoch"]):
                    try:
                        xv,yv,av = move_to_device(next(dl_val_iter), device)
                    except StopIteration:
                        if debug:
                            print(f"Re-initializing validation generator!")
                        dl_val_iter = iter(dl_val)
                        xv,yv,av = move_to_device(next(dl_val_iter), device)
                    wv,hv,sv,siv,initv = xv
                    yv, = yv
                    pv = model(wv, hv, sv, siv, yv, device=device)

                    for mk,metric in metrics.items():
                        tmpm = metric(pv,yv)
                        tmpm_np = tmpm.cpu().detach().numpy()
                        metrics_batch["val"][mk][-1].append(tmpm_np)

                    ## update the sample tracking evaluator
                    a_d,a_s,t = av
                    ess_v.add_batch({
                        "aux-static":a_s.cpu().detach().numpy(),
                        "time":t.cpu().detach().numpy()
                        })

                ## Aggregate batch-wise metrics for simpler epoch-wise stats
                for mk in metrics_batch["val"].keys():
                    metrics_epoch["val"][mk].append([
                        float(np.average(metrics_batch["val"][mk][-1])),
                        float(np.std(metrics_batch["val"][mk][-1])),
                        ])

            ## check for early stopping given mean error of all batches in
            ## the most recent epoch
            early_stop = stopper.early_stop(metrics_epoch["val"][lm][-1][0])

        ## if the model is collecting samples, save them to a pkl in model dir
        if config["model"]["args"]["sample_retain_frequency"]:
            pkl.dump(
                move_to_device(model.samples, "cpu"),
                model_dir.joinpath(f"{config['name']}_samples.pkl").open("wb")
                )

        ## update the batch-wise metric pkl
        pkl.dump(metrics_batch, metric_pkl_path.open("wb"))

        ## update the epoch-wise metrics json
        json.dump(metrics_epoch, metric_json_path.open("w"), indent=2)
        if early_stop:
            print(f"Early stop triggered!")
            break

        ## Save the sample tracking pkls; de-indent this when functional
        ess_t.to_pkl(model_dir.joinpath(ess_t.meta["name"]+".pkl"))
        ess_v.to_pkl(model_dir.joinpath(ess_v.meta["name"]+".pkl"))
