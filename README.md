# ResNeSt-MegEngine

The MegEngine Implementation of ResNeSt.

## Usage

Install dependency.

```bash
pip install -r requirements.txt
```

Convert trained weights from torch to megengine, the converted weights will be save in ./pretained/

```bash
python convert_weights.py -m resnest50
```

Import from megengine.hub:

Way 1:

```python
from  megengine import hub
modelhub = hub.import_module(repo_info='asthestarsfalll/resnest-megengine', git_host='github.com')

# load ResNeSt model and custom on you own
resnest = modelhub.ResNet(modelhub.Bottleneck, [3, 24, 36, 3], radix=2, groups=1, bottleneck_width=64,
                          deep_stem=True, stem_width=64, avg_down=True, avd=True, avd_first=False)

# load pretrained model 
pretrained_model = modelhub.resnest50(pretrained=True) 
```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'resnest50'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/resnest-megengine', entry=model_name, git_host='github.com', pretrained=True)
```

Currently only support resnest50, you can run convert_weights.py to convert other models.
For example:

```bash
  python convert_weights.py -m resnest101
```

Then load state dict manually.

```python
model = modelhub.convnext_small()
model.load_state_dict(mge.load('./pretrained/resnest101.pkl'))
# or
model_name = 'resnest101'
model = hub.load(
    repo_info='asthestarsfalll/resnest-megengine', entry=model_name, git_host='github.com')
model.load_state_dict(mge.load('./pretrained/resnest101.pkl'))
```

## TODO

- [ ] add train codes
- [ ] maybe export to some inference framwork

## Reference

[The official implementation of ResNeSt](https://github.com/zhanghang1989/ResNeSt)
