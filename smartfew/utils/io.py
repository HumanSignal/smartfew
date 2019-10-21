import os
from appdirs import user_data_dir, user_cache_dir


_APP_NAME = 'smartfew'


def get_data_dir():
    data_dir = user_data_dir(appname=_APP_NAME)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_cache_dir():
    cache_dir = user_cache_dir(appname=_APP_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
