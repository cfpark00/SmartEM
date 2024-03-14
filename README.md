# SmartEM
SmartEM: Machine Learning Guided Electron Microscopy Acquisition for Faster Connectomics

# Installation
Assuming you have python3.10 and git installed:

```
git clone https://github.com/cfpark00/SmartEM.git
cd SmartEM
python3.10 -m venv SmartEM_env
source SmartEM_env/bin/activate
pip install -e .
```


# Status

- examples/smart_em_script.py: SmartEM script for running the SmartEM pipeline, Commented

| File     | Commented |
| -------- | --------- |
| examples/smart_em_script.py  | YES       |
| examples/smartem/smartem.py  | YES       |
| examples/smartem/tools.py  | YES       |
| smartem/online/microscope.py  | NO        |
| smartem/online/get_rescan_maps.py | NO        |
| smartem/online/models/UNet.py | NO        |
| smartem/offline    | NO        |