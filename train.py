import pytorch_lightning as pl
import torch
import argparse
import det2d.dataset.load_datasets
from det2d.model import create_model
from det2d.dataset import load_dataset
import yaml
import matplotlib.pyplot as plt
plt.ioff()


def parse_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--config', help='train config file path')
    args = parser.parse_args()
    return args

def main(args):
    config_file = args.config
    f = open(config_file, 'r')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = create_model(cfg['model'])
    train_dataloader, val_dataloader = load_dataset(cfg['data'])

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)



if __name__ == '__main__':
    args = parse_args()
    main(args)


