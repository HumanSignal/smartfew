import os
import logging
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from appdirs import user_data_dir
from torch.utils.tensorboard import SummaryWriter

_PROTONET_CHECKPOINT = 'protonet.pt'


logger = logging.getLogger(__name__)


class ProtonetEncoder(nn.Module):

    def __init__(self, input_dim, layers=(100, 100), dropout=0):
        super(ProtonetEncoder, self).__init__()
        self.input_dim = input_dim
        self.layers = layers
        self.dropout = dropout

        l = []
        n_inputs = input_dim
        for i, layer_dim in enumerate(self.layers):
            l.append(self._layer(
                n_inputs=n_inputs,
                n_outputs=layer_dim,
                last_layer=(i == len(layers) - 1)
            ))
            n_inputs = layer_dim
        self._model = nn.Sequential(*l)

    def _layer(self, n_inputs, n_outputs, last_layer=False):
        layers = [nn.Linear(in_features=n_inputs, out_features=n_outputs)]
        if not last_layer:
            layers.append(nn.ReLU())
            if self.dropout:
                layers.append(nn.Dropout(p=self.dropout))
        if len(layers) == 1:
            return layers[0]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class Protonet(nn.Module):

    def __init__(self, input_dim, layers_dim, dropout=0., use_gpu=False):
        super(Protonet, self).__init__()
        self.use_gpu = use_gpu
        self.input_dim = input_dim
        self.layers_dim = layers_dim
        self.dropout = dropout
        self._encoder = self._on_gpu(ProtonetEncoder(input_dim, layers_dim, dropout))

    def _on_gpu(self, *args):
        out = []
        for tensor in args:
            if self.use_gpu:
                out.append(tensor.cuda())
            else:
                out.append(tensor)
        if len(out) == 1:
            return out[0]
        return out

    def _pairwise_distances(self, x, y):
        x_norm = torch.sum(x * x, 1)[:, None]
        y_norm = torch.sum(y * y, 1)[:, None]

        dist = x_norm + y_norm.permute(1, 0) - 2.0 * torch.mm(x, y.permute(1, 0))
        dist = torch.clamp(dist, min=1e-8)
        return torch.sqrt(dist)

    def forward(self, samples, classes, unique_classes, num_classes_per_episode, num_support_per_class, num_query_per_class):
        random_classes = np.random.choice(unique_classes, num_classes_per_episode, replace=False).astype(int)
        query_samples = self._on_gpu(torch.Tensor())
        query_classes = []

        prototypes = []
        for target_class_idx, target_class in enumerate(random_classes):
            samples_per_class = samples[(classes == target_class).nonzero().long().view(-1)]
            perm_idx = torch.randperm(samples_per_class.shape[0])
            support_samples_per_class, query_samples_per_class = self._on_gpu(
                samples_per_class[perm_idx[:num_support_per_class]],
                samples_per_class[perm_idx[num_support_per_class:num_support_per_class + num_query_per_class]]
            )
            support_encodings_per_class = self._encoder(support_samples_per_class)
            prototype = torch.sum(support_encodings_per_class, 0).unsqueeze(1).transpose(0, 1) / num_support_per_class
            prototypes.append(prototype)
            query_samples = torch.cat((query_samples, query_samples_per_class), 0)
            query_classes.extend([target_class_idx] * query_samples_per_class.shape[0])

        query_classes = self._on_gpu(torch.Tensor(query_classes).long())
        prototypes = self._on_gpu(torch.cat(prototypes))
        query_encodings = self._encoder(query_samples)
        query_dists = self._pairwise_distances(query_encodings, prototypes)

        return -query_dists, query_classes


def train_model(
    samples, classes, checkpoint_dir=None, warm_start=True, use_gpu=False, num_episodes=1000, optimizer='Adam',
    layers=(100, 100), dropout=0.2, num_classes_per_episode=2, num_support_per_class=10, num_query_per_class=10
):

    model = Protonet(
        input_dim=samples.shape[1],
        layers_dim=layers,
        use_gpu=use_gpu,
        dropout=dropout
    )

    if optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    elif optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    else:
        raise ValueError(f'Unknown optimizer {optimizer}')

    if warm_start:
        load_checkpoint(model, optim, checkpoint_dir)

    unique_classes = np.unique(classes)
    samples = torch.from_numpy(samples).float()
    classes = torch.from_numpy(classes).float()

    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    for i in tqdm(range(num_episodes)):
        optim.zero_grad()
        logits, labels = model(
            samples, classes, unique_classes,
            num_classes_per_episode, num_support_per_class, num_query_per_class
        )
        loss_value = loss(logits, labels)
        writer.add_scalar('Loss', loss_value.item(), i)
        loss_value.backward()
        optim.step()

    save_checkpoint(model, optim, checkpoint_dir)


def load_model(checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = user_data_dir(__name__)
    checkpoint_file = os.path.join(checkpoint_dir, _PROTONET_CHECKPOINT)
    if not os.path.exists(checkpoint_file):
        logger.warning(f'Can\'t load model state from checkpoint {checkpoint_file}: file doesn\'t exist')
        return

    checkpoint = torch.load(checkpoint_file)
    model = Protonet(input_dim=checkpoint['input_dim'], layers_dim=checkpoint['layers_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def apply_model(samples, model, classes=None):
    encodings = model._encoder(torch.from_numpy(samples).float()).detach().numpy()
    if classes is None:
        return encodings
    agg_encodings = []
    unique_classes = np.unique(classes)
    for target_class in unique_classes:
        agg_encodings.append(np.mean(encodings[np.where(classes == target_class)[0]], axis=0))
    encodings = np.vstack(agg_encodings)
    return encodings, unique_classes


def save_checkpoint(model, optimizer, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = user_data_dir(__name__)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'input_dim': model.input_dim,
        'layers_dim': model.layers_dim,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint_file = os.path.join(checkpoint_dir, _PROTONET_CHECKPOINT)
    torch.save(checkpoint, checkpoint_file)
    logger.info(f'Checkpoint saved to {checkpoint_file}')


def load_checkpoint(model, optimizer, checkpoint_dir=None, for_inference=False):
    if checkpoint_dir is None:
        checkpoint_dir = user_data_dir(__name__)

    checkpoint_file = os.path.join(checkpoint_dir, _PROTONET_CHECKPOINT)
    if not os.path.exists(checkpoint_file):
        logger.warning(f'Can\'t load model state from checkpoint {checkpoint_file}: file doesn\'t exist')
        return

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    if for_inference:
        model.eval()
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
    logger.info(f'Checkpoint loaded from {checkpoint_file} for {"inference" if for_inference else "train"}')
