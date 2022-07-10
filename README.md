# Triton Python model
The goal of Triton Python model package `triton-python-model` is to streamline Python model serving in [Visual Data Processing (VDP) project](https://github.com/instill-ai/vdp).

## Use custom python execution environments in Triton
From [Triton python backend](https://github.com/triton-inference-server/python_backend/tree/main#using-custom-python-execution-environments) guideline:

> Python backend uses a stub process to connect your model.py file to the Triton C++ core. This stub process has an embedded Python interpreter with a fixed Python version.

We maintain Dockerfiles for Instill AI's official [conda](https://docs.conda.io/en/latest/) environment for Triton python-backend. In the conda environment, we use Python 3.8 and install packages
- [scikit-learn==1.1.1](https://github.com/scikit-learn/scikit-learn)
- [Pillow==9.1.1](https://github.com/python-pillow/Pillow)
- [PyTorch==1.11.0](https://github.com/pytorch/pytorch)
- [torchvision=0.12.0](https://pytorch.org/vision/stable/index.html)
- triton-python-model

Please read [here](https://github.com/pytorch/pytorch/wiki/PyTorch-Versions) for the PyTorch compatible domain libraries.

## License

See the [LICENSE](https://github.com/instill-ai/triton-python-model/blob/main/LICENSE) file for licensing information.
