# SmartEM
SmartEM: Machine Learning Guided Electron Microscopy Acquisition for Faster Connectomics

# Installation
Assuming you have python3.10 and git installed:

```
git clone https://github.com/cfpark00/SmartEM.git
cd SmartEM
python3.10 -m venv SmartEM_env
source SmartEM_env/bin/activate
```

Now install pytorch and torchvision from the [official website](https://pytorch.org/get-started/locally/)

Now run:
```
pip install -e .
```

Note: Some parts of this package, including the `ThermoFisherVerios` class which helps implement this method on a Thermo Fisher scanning electron microscope depend on the proprietary package `autoscript_sdb_microscope_client`.

## Example Installation using venv on Windows

1. Create virtual environment using venv with python 3.10
```
<Path to python 3.10 executable e.g. C:\Users\Support\AppData\Local\Programs\Python\Python310\python.exe> -m venv <venv name e.g. venv_310>
```
2. Activate environment
```
<venv name e.g. venv_310>\Scripts\activate
```
3. Upgrade pip
```
python -m pip install --upgrade pip
```
4. Go to code directory
```
cd <path to SmartEM>
```
5. Install code in editable mode
```
python -m pip install -e .
```
6. Check that smartem was installed (optional)
```
pip list
```
7. Run example script
```
python examples\smart_em_script.py
```
Note: we use `python -m` in order to call `pip` through our environment's interpreter. If you omit this, you might be running a different version of `pip` that exists outside your environment.
# Status

- examples/smart_em_script.py: SmartEM script for running the SmartEM pipeline
