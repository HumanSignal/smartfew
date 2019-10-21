from setuptools import setup, find_packages


setup(
    name='smartfew',
    version='0.0.1',
    description='SmartFew Tool',
    author='Nikolai Liubimov',
    author_email='nik@heartex.net',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask==1.1.1',
        'torch==1.3.0',
        'torchvision==0.4.1'
    ],
)