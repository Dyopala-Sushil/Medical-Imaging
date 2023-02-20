import torch
from tqdm import tqdm
import os
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import mlflow
from monai.utils import  set_determinism


def validate(model, loss_function, val_loader, dice_metric, metric_values, post_pred, post_label, device):
    model.eval()    # Setting model in eval mode
    with torch.no_grad():
        step = 0
        val_loss = 0
        for val_data in val_loader:
            step += 1
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
        
            roi_size = (160, 160, 160)
            sw_batch_size = 4

            val_outputs = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model)
            
            val_l = loss_function(val_outputs, val_labels)
            val_loss += val_l

            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            

            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result using reduction
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

        metric_values.append(metric)

        val_loss /= step

    return metric, val_loss


def train(model, loss_function, optimizer, scheduler, dice_metric, train_loader, val_loader, post_pred, 
          post_label, device, save_dir, start_epoch=0, end_epoch=10, val_interval=2, 
          checkpoint_path="", load_model=False):
  
    """
    Train and Evaluate the model

    Args:
        model: model from MONAI networks.net
        loss_function: instance of monai.losses
        optimizer: instance of torch.optim
        train_loader: loads training sets in batch
        val_loader: loads validation set in batch
        post_pred: monai transform to post process the predictions through argmax and one-hot encoding
        post_label: monai transfrom to post process the label to one-hot format
        device: str
        save_dir: path to directory where checkpoint is saved
        start_epoch: epoch number to start from if to resume from previous checkpoint 
        end_epoch: epoch number to stop training
        val_interval: interval at which model is validated across val set
        checkpoint_path: str path to previous checkpoint from which training is to be resumed
        load_model: bool, whether to load model from previous checkpoint or not

    """
    set_determinism(seed=0)

    if load_model:
        checkpoint = torch.load(checkpoint_path)
        best_metric = checkpoint["best_metric"]
        best_metric_epoch = checkpoint["best_metric_epoch"]
        train_loss_values = checkpoint["train_loss"]
        metric_values = checkpoint["val_dice"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        checkpoint = {}
        best_metric = -1
        best_metric_epoch = -1
        train_loss_values = []
        metric_values = []
    
   
    for epoch in range(start_epoch, end_epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{end_epoch}")
        model.train()   # set model on training mode
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            train_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            train_labels = [post_label(i) for i in decollate_batch(labels)]
            

            # compute metric for current iteration
            dice_metric(y_pred=train_outputs, y=train_labels)

            train_metric = dice_metric.aggregate().item()

        lr = scheduler.get_last_lr()[0]
        mlflow.log_metric("lr", lr, step=epoch+1)

        scheduler.step()    
        # reset the status for next validation round
        dice_metric.reset()
        mlflow.log_metric("train_dice_score", train_metric, step=epoch+1)

            
        epoch_loss /= step
        train_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            metric, val_loss = validate(model, loss_function, val_loader, dice_metric, metric_values, post_pred, post_label, device)
            metric_values.append(metric)
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    save_dir, "best_metric_model.pth"))
                print("Saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
            mlflow.log_metric("val_dice_score", metric, step=epoch+1)
            mlflow.log_metric("val_dice_loss", val_loss, step=epoch+1)

        mlflow.log_metric("train_dice_loss", epoch_loss, step=epoch+1)

    
    mlflow.log_metric("best_dice_score", best_metric)
    mlflow.log_metric("best_dice_epoch", best_metric_epoch)

    checkpoint["train_loss"] = train_loss_values
    checkpoint["val_dice"] = metric_values
    checkpoint["best_metric_epoch"] = best_metric_epoch
    checkpoint["best_metric"] = best_metric
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, os.path.join(save_dir, "my_checkpoint1.pth.tar"))

    return model
