from setuptools import setup

setup(name='AutoPower',
      version='0.0',
      description='AutoPower',
      url='https://github.com/florpi/AutoPower',
      install_requires=['camb',
                        'h5py',
                        'jupyter',
                        'matplotlib',
                        'numpy',
                        'scikit-learn',
                        'scipy',
                        'seaborn',
                        'torch',
                        'tqdm'],
      packages=['hsr4hci'],
      zip_safe=False)
