# Neural Style Transfer

Implementation of Neural Style Transfer using PyTorch Lightning.

## Setup

To run the project you need to install all dependencies specified in `requirements.txt`, you can also setup a conda environment. 
If you need to update requirements, edit the `requirements.in` file, then run:

```sh
pip-compile requirements.in -v --find-links=https://download.pytorch.org/whl/torch_stable.html
```

### Set up the conda environment

With conda first you need to create an environment, as defined in `environment.yml`:

```sh
conda env create -f environment.yml
```

After running `conda env create`, activate the new environment and install the requirements (on Windows might need to add `wincertstore` to requirements):

```sh
conda activate neural-style-transfer
pip-sync requirements.txt
```

## Running the Project

To run execute a command:

```sh
python -m style_transfer.style_transfer "/path/to/content/image" "/path/to/style/image"
```

This command will perform style transfer with default options and save the result in data/results.

To see additional options add `--help` flag. For example, adding `--gpu` will make the program use gpu during transfer if available.


## Requirements

* NumPy
* OpenCV
* PyTorch
* torchvision
* PyTorch Lightning


## References

* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* [Neural Transfer using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
* [Neural Style Transfer](https://keras.io/examples/generative/neural_style_transfer/)
