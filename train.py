from tqdm import tqdm
from utils import *
from models import *
from criterion import *
from dataloader2 import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
from torch.utils.data import DataLoader
import wandb
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:

        target = target.squeeze(dim=1).long()
        input = input.to(DEVICE)
        target = target.to(DEVICE)

        outputs = model(input)
        loss = criterion(outputs, target.long())
        dice = dice_score(outputs.clone().detach(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['dice'].update(dice.item(), input.size(0))
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    if config["plot_training_image"]:
        label = target[0].detach().clone()
        output = outputs[0].detach().clone()
        label = label.cpu().numpy()
        label = mask_to_rgb(label)
        output = output.softmax(dim=0)
        output = output.cpu().numpy()
        output = np.argmax(output, axis=0)
        output = mask_to_rgb(output)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(label)
        axs[0].set_title('Label')
        axs[1].imshow(output)
        axs[1].set_title('Output')
        plt.show()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('dice', avg_meters['dice'].avg)
    ])


def validate(config, val_loader, model, criterion):
    avg_meters = {"loss": AverageMeter(),
                  "dice": AverageMeter()}

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:

            target = target.squeeze(dim=1).long()
            input = input.to(DEVICE)
            target = target.to(DEVICE)


            outputs = model(input)
            loss = criterion(outputs, target)
            dice = dice_score(outputs.clone().detach(), target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['dice'].update(dice.item(), input.size(0))
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    if config["plot_training_image"]:
        label = target[0].detach().clone()
        output = outputs[0].detach().clone()
        label = label.cpu().numpy()
        label = mask_to_rgb(label)
        output = output.softmax(dim=0)
        output = output.cpu().numpy()
        output = np.argmax(output, axis=0)
        output = mask_to_rgb(output)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(label)
        axs[0].set_title('Label')
        axs[1].imshow(output)
        axs[1].set_title('Output')
        plt.show()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('dice', avg_meters['dice'].avg)
    ])
def initialize(config):

    model = UNet(config['num_classes'],config['input_channels'])
    weights_init(model)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    else:
        raise NotImplementedError

    if config['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().to(DEVICE)
    elif config['loss'] == 'CEDiceLoss':
        unormed_weight = torch.Tensor([[0.05, 0.4, 0.45]])
        normed_weight = unormed_weight / torch.sum(unormed_weight)
        weights = normed_weight.to(DEVICE)
        criterion = CEDiceLoss(weights).to(DEVICE)
    else:
        raise NotImplementedError

    return model.to(DEVICE), optimizer, scheduler, criterion.to(DEVICE)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hello')

    # Add arguments
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--plot_training_image', action='store_true', help='Plot image for each epoch')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer choice')
    parser.add_argument('--backbone', type=str, default='resnet152', help='Backbone of model architecture')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--loss', type=str, default='CEDiceLoss', help='Loss function')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', help='Learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--early_stopping', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/model.pth', help='Path to save checkpoints')
    parser.add_argument('--checkpoint_end_epoch_path', type=str, default='checkpoint/model_end.pth', help='Path to save checkpoints at the end')

    args = parser.parse_args()

    return args

def main():

    IMAGE_TRAINING_PATH = "dataset/train/train"
    IMAGE_GT_PATH = "dataset/train_gt/train_gt"

    CHECKPOINT_DIR = "checkpoint"

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    config = vars(parse_arguments())


    train_loss_array = list()
    test_loss_array = list()
    best_loss = 1e9
    use_wandb = True
    trigger = 0

    if use_wandb:
        wandb.login(
            key="34c6ba04b2b20a97743debee3dda601a31640975",
        )
        wandb.init(
            project="BKAI-IGH-NeoPolyp"
        )

    all_dataset = SplitDataset(training_images_path=IMAGE_TRAINING_PATH,
                               training_masks_path=IMAGE_GT_PATH)

    train_set, valid_set = all_dataset.create_training_set()

    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True)

    model, optimizer, scheduler, criterion = initialize(config)

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, train_dataloader, model, criterion, optimizer)
        val_log = validate(config, valid_dataloader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        trigger += 1

        if val_log["loss"] < best_loss:
            checkpoint = {
                "model": model.state_dict(),
                "backbone": config["backbone"],
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "num_classes": config["num_classes"],
                "input_channels": config["input_channels"]
            }

            torch.save(checkpoint, config["checkpoint_path"])

            best_loss = val_log["loss"]
            print("=> saved best model")
            trigger = 0

        torch.cuda.empty_cache()

        train_loss_array.append(train_log["loss"])
        test_loss_array.append(val_log["loss"])

        if use_wandb:
            wandb.log({"Train loss": train_log["loss"],
                       "Valid loss": val_log["loss"],
                       "Valid dice coefficient": val_log["dice"]})

        torch.cuda.empty_cache()

    checkpoint = {
        "model": model.state_dict(),
        "backbone": config["backbone"],
        "optimizer": optimizer.state_dict(),
        "epoch": -1,
        "num_classes": config["num_classes"],
        "deep_supervision": config["deep_supervision"],
        "input_channels": config["input_channels"]
    }

    torch.save(checkpoint, config["checkpoint_end_epoch_path"])

    print(trigger)  # if trigger == 0 then checkpoint_path == checkpoint_end_epoch_path

    # zip pretrained weight file
    zip_end_checkpoint = config["checkpoint_end_epoch_path"].rsplit(".", 1)[0] + ".zip"
    create_zip_file(file_to_zip=config["checkpoint_end_epoch_path"], zip_file_name=zip_end_checkpoint)

    print(f"Save check point at: {zip_end_checkpoint}")

    zip_checkpoint = config["checkpoint_path"].rsplit(".", 1)[0] + ".zip"
    create_zip_file(file_to_zip=config["checkpoint_path"], zip_file_name=zip_checkpoint)

    print(f"Save check point at: {zip_checkpoint}")




if __name__ == "__main__":
    main()
