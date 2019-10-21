import requests
import os
import hashlib
import io
import logging
import torch

from torch import nn
from torchvision import models, transforms
from appdirs import user_cache_dir
from PIL import Image


logger = logging.getLogger(__name__)


class ImageEncoder(object):

    def __init__(self, for_train=True, model_name='resnet50', cache_dir=None):
        self.for_train = for_train
        self.model_name = model_name
        self.cache_dir = cache_dir if cache_dir is not None else user_cache_dir(__name__)
        os.makedirs(self.cache_dir, exist_ok=True)

        self._preprocessing = self._get_preprocessing(for_train)
        self._encoder = None
        if model_name == 'resnet50':
            self._encoder = self._resnet50()
        else:
            raise NotImplementedError(f'Unknown model name "{model_name}"')

    def _get_preprocessing(self, for_train=True):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if for_train:
            return transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    def _resnet50(self):
        model = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        model.eval()
        return model

    def _load_from_file(self, filepath):
        image_data = Image.open(filepath).convert('RGB')
        return image_data

    def _load_from_url(self, url):
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
        except Exception as exc:
            logger.error(f'Failed downloading {url}. Reason: {exc}', exc_info=True)
            return
        with io.BytesIO(r.content) as f:
            return Image.open(f).convert('RGB')

    def _load_from_url_cached(self, url):
        filename = hashlib.md5(url.encode()).hexdigest()
        filepath = os.path.join(self.cache_dir, filename)
        if not os.path.exists(filepath):
            try:
                r = requests.get(url)
                r.raise_for_status()
                with io.open(filepath, mode='wb') as fout:
                    fout.write(r.content)
            except Exception as exc:
                logger.error(f'Failed downloading {url} to {filepath}. Reason: {exc}', exc_info=True)
                return
        return self._load_from_file(filepath)

    def encode(self, items, is_urls=True):
        preprocessed_images = []
        for item in items:
            if is_urls:
                data = self._load_from_url_cached(item)
            else:
                data = self._load_from_file(item)
            preprocessed_images.append(self._preprocessing(data))
        preprocessed_images = torch.stack(preprocessed_images)
        with torch.no_grad():
            e = self._encoder(preprocessed_images)
            tensor_images = torch.reshape(e, (e.size(0), e.size(1)))
            return tensor_images.data.numpy()
