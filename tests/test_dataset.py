from det2d.dataset import load_dataset
import yaml
import matplotlib.pyplot as plt
plt.ioff()
from torch.utils.data import DataLoader

file = './cfg/dataset_test.yaml'
f = open(file, 'r')
cfg = yaml.load(f, Loader=yaml.FullLoader)
train_dataset, val_dataset = load_dataset(cfg['data'])
for i in enumerate(val_dataset):
    pass

print(cfg)

