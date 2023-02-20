from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd
)
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.utils import  set_determinism
from prepare_data import prepare_data
from process_data import get_testloader, get_trainloader
from model import getModel
from train import train
import math
import torch
import mlflow
import os
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split


if __name__=="__main__":
    PATH2DATA = "Data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_determinism(seed=0)
    data_dicts = prepare_data(PATH2DATA)
    val_interval = 1
    # Set the validation set proportion
    val_size = 0.2

    # Splitting dataset into train and validation set
    train_files, val_files = train_test_split(data_dicts, test_size=val_size, random_state=42)
    # Print number of samples in train and validation sets
    print(f"Number of samples in train set: {len(train_files)}")
    print(f"Number of samples in validation set: {len(val_files)}")

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,  
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )

    train_loader = get_trainloader(train_files, train_transforms)
    val_loader = get_testloader(val_files, val_transforms)

    model = getModel(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120,160], gamma=0.5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")    

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    # Directory where checkpoint will be saved
    experiment_name = "Experiment1-Val-Score-Monitoring"
    run_name="run7-multistepLR-valsize-dec"
    experiment_dir = "results/" + experiment_name
    save_dir = "results/" + experiment_name + "/" + run_name

    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("train_batchsize", 2)
        mlflow.log_param("val_batchsize", 1)
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("val_interval", val_interval)
        mlflow.log_param("gamma_lr", 0.5)

        model = train(model, loss_function, optimizer, scheduler, dice_metric, train_loader, val_loader,
                    post_pred, post_label, device=device, save_dir=save_dir, start_epoch=0, end_epoch=200, val_interval=val_interval,  
                    checkpoint_path="", load_model=False)

       