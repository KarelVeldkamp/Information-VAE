from model import *
from data import *
from helpers import *
import numpy as np
import pandas as pd
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import time
import os

# set working directory to source file location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# set device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load configurations
with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']



theta=np.random.normal(0,1,cfg['N']*cfg['mirt_dim']).reshape((cfg['N'], cfg['mirt_dim']))
Q = pd.read_csv(f'parameters/QMatrix{cfg["mirt_dim"]}D.csv', header=None).values

a = np.random.uniform(.5, 2, Q.shape[0] * cfg['mirt_dim']).reshape((Q.shape[0], cfg['mirt_dim']))  # draw discrimination parameters from uniform distribution
a *= Q
b = np.linspace(-2, 2, Q.shape[0], endpoint=True)  # eqally spaced values between -2 and 2 for the difficulty
exponent = np.dot(theta, a.T) + b
prob = np.exp(exponent) / (1 + np.exp(exponent))
data = np.random.binomial(1, prob).astype(float)

# introduce missingness
#np.random.seed(cfg['iteration'])
#indices = np.random.choice(data.shape[0]*data.shape[1], replace=False, size=int(data.shape[0]*data.shape[1]*cfg['missing_percentage']))
#data[np.unravel_index(indices, data.shape)] = float('nan')
#data = torch.Tensor(data)

# initialise model and optimizer
logger = CSVLogger("logs", name='simfit', version=0)
trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                  max_epochs=cfg['max_epochs'],
                  enable_checkpointing=False,
                  logger=False,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')])

dataset = SimDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
vae = VAE(nitems=data.shape[1],
            dataloader=train_loader,
            latent_dims=cfg['mirt_dim'],
            hidden_layer_size=cfg['hidden_layer_size'],
            hidden_layer_size2=cfg['hidden_layer_size2'],
            learning_rate=cfg['learning_rate'],
            batch_size=data.shape[0]
)


start = time.time()
trainer.fit(vae)
runtime = time.time()-start

# plot training loss
logs = pd.read_csv(f'logs/simfit/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/simfit/training_loss.png')



