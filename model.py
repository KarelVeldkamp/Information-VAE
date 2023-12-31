import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
from data import *

# Enocder class
class Encoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(Encoder, self).__init__()

        input_layer = nitems

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size2)
        self.densem = nn.Linear(hidden_layer_size2, latent_dims)
        self.denses = nn.Linear(hidden_layer_size2, latent_dims)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """

        # calculate s and mu based on encoder weights
        out = F.elu(self.dense1(x))
        out = F.elu(self.dense2(out))
        mu =  self.densem(out)
        log_sigma = self.denses(out)

        return mu, log_sigma

# Sampling layer
class SamplingLayer(pl.LightningModule):
    def __init__(self):
        super(SamplingLayer, self).__init__()
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        error = self.N.sample(mu.shape)
        # potentially move error vector to GPU
        error = error.to(mu)
        return mu + sigma * error

# Decoder class
class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        # initialise netowrk components
        input_layer = latent_dims
        self.linear = nn.Linear(input_layer, nitems)
        self.activation = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :param m: mask representing which data is missing
        :return: tensor representing reconstructed item responses
        """
        out = self.linear(x)
        out = self.activation(out)
        return out

# Entire variational autoencoder class
class VAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2: int,
                 learning_rate: float,
                 batch_size: int,
                 beta: int = 1):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        """
        super(VAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               latent_dims,
                               hidden_layer_size,
                               hidden_layer_size2
        )

        self.sampler = SamplingLayer()

        self.decoder = Decoder(nitems, latent_dims)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.kl=0

    def forward(self, x: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, log_sigma = self.encoder(x)
        z = self.sampler(mu, log_sigma)
        reco = self.decoder(z)

        # calcualte kl divergence
        kl = 1 + 2 * log_sigma - torch.square(mu) - torch.exp(2 * log_sigma)
        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)
        return reco

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data, mask = batch
        X_hat = self(data)

        # calculate the likelihood, and take the mean of all non missing elements
        # bce = data * torch.log(X_hat) + (1 - data) * torch.log(1 - X_hat)
        # bce = bce*mask
        # bce = -self.nitems * torch.mean(bce, 1)
        # bce = bce / torch.mean(mask.float(),1)

        bce = torch.nn.functional.binary_cross_entropy(X_hat, batch[0], reduction='none')
        bce = torch.mean(bce)  * self.nitems
        bce = bce / torch.mean(mask.float())

        # sum the likelihood and the kl divergence

        #loss = torch.mean((bce + self.encoder.kl))
        loss = bce + self.beta * self.kl
        #self.log('binary_cross_entropy', bce)
        #self.log('kl_divergence', self.encoder.kl)
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def compute_information(self, theta):
        prob = self.decoder(theta)

        # Loop though output dimensions/items, save Fisher information matrix for each
        FIMs = []
        for i in range(prob.shape[0]):
            p = prob[i]
            # compute gradients of log p(x=1) and log p(x=0) to the latent variables
            gradient0 = torch.autograd.grad(torch.log(1-p), theta, retain_graph=True, allow_unused=True)[0]
            gradient1 = torch.autograd.grad(torch.log(p), theta, retain_graph=True, allow_unused=True)[0]

            # Compute the outer product of the gradients with itself
            outer0 = torch.outer(gradient0, gradient0)
            outer1 = torch.outer(gradient1, gradient1)

            # compute the Fisher information matrix by calculating the expectation of this outer product
            FIM = p * outer1 + (1-p) * outer0

            # append to list of FIMs
            FIMs.append(FIM)

        return FIMs


# Entire variational autoencoder class
class VAE_normal(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """

    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 hidden_layer_size2: int,
                 learning_rate: float,
                 batch_size: int,
                 beta: int = 1):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        """
        super(VAE_normal, self).__init__()
        # self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               latent_dims,
                               hidden_layer_size,
                               hidden_layer_size2
                               )

        self.sampler = SamplingLayer()

        self.decoder = Decoder(nitems, latent_dims)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.logscale = nn.Parameter(torch.Tensor([0.0]))
        self.kl = 0

    def forward(self, x: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, log_sigma = self.encoder(x)
        z = self.sampler(mu, log_sigma)
        reco = self.decoder(z)

        # calcualte kl divergence
        kl = 1 + 2 * log_sigma - torch.square(mu) - torch.exp(2 * log_sigma)
        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)
        return reco

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # log prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data, mask = batch
        X_hat = self(data)

        # calculate the likelihood, and take the mean of all non missing elements
        # bce = data * torch.log(X_hat) + (1 - data) * torch.log(1 - X_hat)
        # bce = bce*mask
        # bce = -self.nitems * torch.mean(bce, 1)
        # bce = bce / torch.mean(mask.float(),1)

        ll = self.gaussian_likelihood(X_hat, self.logscale, data)

        # sum the likelihood and the kl divergence

        # loss = torch.mean((bce + self.encoder.kl))
        loss = -ll + self.beta * self.kl
        # self.log('binary_cross_entropy', bce)
        # self.log('kl_divergence', self.encoder.kl)
        self.log('train_loss', loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def compute_information(self, theta):
        pass
