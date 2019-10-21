from setuptools import setup, find_packages


setup(
    name='smartfew',
    version='0.0.1',
    description='SmartFew',
    author='Nikolai Liubimov',
    author_email='nik@heartex.net',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask==1.1.1',
        'requests==2.22.0',
        'appdirs==1.4.3',
        'tqdm==4.36.1',
        'torch==1.3.0',
        'torchvision==0.4.1',
    ],
)