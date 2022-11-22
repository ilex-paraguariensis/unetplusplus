from ...data.loaders.carvana.carvana import CarvanaDataset
from ...trainers.unet_trainer.trainer import train
from ...trainers.unet_trainer.evaluate import test
from ...models.unet.unet_model import UNet
from ...models.unet.pretrained_unet import pretrained_net
import sys

# save_dir = "../save_dir/"
net = UNet(n_channels=3, n_classes=2)
train_dataset = CarvanaDataset(images_dir="./train", masks_dir="./train_masks")
test_dataset = CarvanaDataset(images_dir="./train", masks_dir="./train_masks")

command = sys.argv[1]

if command == "train":
    train(
        net=net,
        dataset=train_dataset,
    )
elif command == "test":
    test(model_class=UNet, dataset=test_dataset)
