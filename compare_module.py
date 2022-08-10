import time
from functools import partial

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF

from models.resnest import Bottleneck, SplAtConv2d
from models.torch_resnest import Bottleneck as TorchBottleneck, SplAtConv2d as TorchSplAtConv2d

GLOBAL_RTOL = 1e-3
BATCH_SIZE = 8

DTYPE_MAPPER = {
    # 'float16': (np.float16, torch.float16),
    'float32': (np.float32, torch.float32),
    # 'float64': (np.float64, torch.float64),
}

def mge_downsample():
    down_layers = []
    stride = 2
    planes = 128
    avg_down = True
    dilation = 1
    if avg_down:
        if dilation == 1:
            down_layers.append(M.AvgPool2d(kernel_size=stride, stride=stride))
        else:
            down_layers.append(M.AvgPool2d(kernel_size=1, stride=1))
        down_layers.append(M.Conv2d(64, planes * 4, kernel_size=1, stride=1, bias=False))
    else:
        down_layers.append(M.Conv2d(64, planes * 4, kernel_size=1, stride=stride, bias=False))
    down_layers.append(M.BatchNorm2d(planes * 4))
    downsample = M.Sequential(*down_layers)
    return downsample


def torch_downsample():
    down_layers = []
    stride = 2
    planes = 128
    avg_down = True
    dilation = 1
    if avg_down:
        if dilation == 1:
            down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=True))
        else:
            down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=True))
        down_layers.append(nn.Conv2d(64, planes * 4, kernel_size=1, stride=1, bias=False))
    else:
        down_layers.append(nn.Conv2d(64, planes * 4, kernel_size=1, stride=stride, bias=False))
    down_layers.append(nn.BatchNorm2d(planes * 4))
    downsample = nn.Sequential(*down_layers)
    return downsample


KWARDS_MAPPER = {
    "Bottleneck": [
        dict(inplanes=64, planes=16, stride=1, downsample=None,
             radix=2, cardinality=1, bottleneck_width=64, avd=True,
             avd_first=True, dilation=1, is_first=True,
             dropblock_prob=0.0, last_gamma=False),
        # dict(inplanes=64, planes=64, stride=1, downsample=None,
        #      radix=1, cardinality=1, bottleneck_width=64, avd=True,
        #      avd_first=True, dilation=1, is_first=False,
        #      dropblock_prob=0.0, last_gamma=False),
        # dict(inplanes=64, planes=64, stride=1, downsample=None,
        #      radix=1, cardinality=1, bottleneck_width=64, avd=True,
        #      avd_first=False, dilation=2, is_first=False,
        #      dropblock_prob=0.0, last_gamma=False),
        # dict(inplanes=64, planes=64, stride=1, downsample=None,
        #      radix=1, cardinality=1, bottleneck_width=64, avd=True,
        #      avd_first=False, dilation=4, is_first=False,
        #      dropblock_prob=0.0, last_gamma=False),
        # dict(inplanes=64, planes=64, stride=1, downsample=None,
        #      radix=1, cardinality=1, bottleneck_width=64, avd=False,
        #      avd_first=False, dilation=1, is_first=False,
        #      dropblock_prob=0.0, last_gamma=False),
        dict(inplanes=64, planes=128, stride=2, downsample=None,
             radix=2, cardinality=1, bottleneck_width=64, avd=True,
             avd_first=True, dilation=1, is_first=True,
             dropblock_prob=0.0, last_gamma=False),
    ],
    "SplAtConv2d": [
        dict(in_channels=128, channels=64, kernel_size=3, stride=(1, 1),
             padding=(0, 0), dilation=(1, 1), groups=1, bias=True,
             radix=2, reduction_factor=4),
        dict(in_channels=128, channels=64, kernel_size=3, stride=(1, 1),
             padding=(0, 0), dilation=(1, 1), groups=1, bias=True,
             radix=2, reduction_factor=4, norm_layer=None)
    ]
}


CLASS_MAPPER = {
    # "SplAtConv2d": (SplAtConv2d, TorchSplAtConv2d),
    "Bottleneck": (Bottleneck, TorchBottleneck),
}


def generate_inputs(shape, dtype='float32'):
    inp = np.random.randn(*shape)
    types = DTYPE_MAPPER[dtype]
    mge_inp = mge.tensor(inp, dtype=types[0])
    torch_inp = torch.tensor(inp, dtype=types[1])
    return mge_inp, torch_inp


def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module


def convert_state_dict(torch_module, torch_dict):
    mge_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_module, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        mge_dict[k] = data

    return mge_dict


def is_in_string(targets: list, s: str):
    return any(t in s for t in targets)


def convert_dtype(m):
    pass


def test_func(mge_tensor, torch_tensor):
    mge_out = mge_tensor.numpy()
    if torch.cuda.is_available():
        torch_out = torch_tensor.detach().cpu().numpy()
    else:
        torch_out = torch_tensor.detach().numpy()
    result = np.isclose(mge_out, torch_out, rtol=GLOBAL_RTOL)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def get_channels(kwards):
    for n in ['inplanes', 'in_channels']:
        if n in kwards:
            ch = kwards[n]
            if isinstance(ch, list):
                return ch
            return [ch]
    else:
        # if 'dims' in kwards:
        #     return [3]
        return list(np.random.randint(1, 2048, size=[1]))


def main():
    print(f"Begin test with rtol = {GLOBAL_RTOL}, batch size ={BATCH_SIZE}")
    print()
    unalign_list = []
    a = 0
    for k, (mge_class, torch_class) in CLASS_MAPPER.items():
        kwards = KWARDS_MAPPER.get(k, [{}])
        print(f"Begin test {k}:")
        for kw in kwards:
            print(f"\t with kwards {kw}:")
            if a == 1:
                kw['downsample'] = mge_downsample()
            mge_module = mge_class(**kw)
            mge_module.eval()
            if a==1:
                kw['downsample'] = torch_downsample()
            a += 1
            torch_module = torch_class(**kw)
            torch_module.eval()
            channels = get_channels(kw)
            # for sp_dim in [64, 224, 512, 1024]:
            for sp_dim in [56]:
                input_shape = (BATCH_SIZE, *channels, sp_dim, sp_dim)
                for dtype in DTYPE_MAPPER.keys():
                    mge_inp, torch_inp = generate_inputs(input_shape, dtype)
                    print(f"\t\t with shape {mge_inp.shape}:")
                    print(f"\t\t\t with dtype {dtype}:")
                    torch_dict = torch_module.state_dict()
                    mge_dict = convert_state_dict(torch_module, torch_dict)
                    mge_module.load_state_dict(mge_dict)

                    st = time.time()
                    mge_out = mge_module(mge_inp)
                    mge_time = time.time() - st

                    st = time.time()
                    torch_out = torch_module(torch_inp)
                    torch_time = time.time() - st
                    if isinstance(mge_out, list):
                        for idx, (m, t) in enumerate(zip(mge_out, torch_out)):
                            print(f'The {i+1}th element of output:')
                            ratio, allclose, abs_err, std_err = test_func(m, t)
                            print(
                                f"\t\t\t\tResult: {allclose}, {ratio*100 : .4f}% elements is close enough\n \t\t\t\t which absolute error is  {abs_err} and absolute std is {std_err}")
                    elif isinstance(mge_out, dict):
                        for k in mge_out.keys():
                            print(f"Key: {k}")
                            m = mge_out[k]
                            t = torch_out[k]
                            ratio, allclose, abs_err, std_err = test_func(m, t)
                            print(
                                f"\t\t\t\tResult: {allclose}, {ratio*100 : .4f}% elements is close enough\n \t\t\t\t which absolute error is  {abs_err} and absolute std is {std_err}")
                    else:
                        ratio, allclose, abs_err, std_err = test_func(
                            mge_out, torch_out)
                        print(
                            f"\t\t\t\tResult: {allclose}, {ratio*100 : .4f}% elements is close enough\n \t\t\t\t which absolute error is  {abs_err} and absolute std is {std_err}")
                    if not allclose:
                        unalign_list.append(k)
                    print(
                        f"\t\t\t\ttime used: megengine: {mge_time : .4f}s, torch: {torch_time : .4f}s")
    print(f"Test down, unaligned module: {list(set(unalign_list))}")


if __name__ == "__main__":
    a = M.Conv2d(1, 1, 1)
    main()
