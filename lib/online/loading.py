import torch
import os
import sys
from pathlib import Path
import importlib
from online.model_constructor import NetConstructor

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(missing_keys))

    print('=========================================')
    # clean it to no batch_tracked key words
    unused_pretrained_keys = [k for k in unused_pretrained_keys if 'num_batches_tracked' not in k]

    print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def load_pretrain(model, pretrained_dict):

    device = torch.cuda.current_device()

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_network(ckpt_path=None, constructor_fun_name='online_resnet18', constructor_module='lib.models.online.bbreg.online'):

        # Load network
        checkpoint_dict = torch.load(ckpt_path) # key: net

        # get model structure from constructor
        net_constr = NetConstructor(fun_name=constructor_fun_name, fun_module=constructor_module)
        # Legacy networks before refactoring

        net = net_constr.get()

        net = load_pretrain(net, checkpoint_dict['net'])

        return net


def load_weights(net, path, strict=True):
    checkpoint_dict = torch.load(path)
    weight_dict = checkpoint_dict['net']
    net.load_state_dict(weight_dict, strict=strict)
    return net


def torch_load_legacy(path):
    """Load network with legacy environment."""

    # Setup legacy env (for older networks)
    _setup_legacy_env()

    # Load network
    checkpoint_dict = torch.load(path)

    # Cleanup legacy
    _cleanup_legacy_env()

    return checkpoint_dict


def _setup_legacy_env():
    importlib.import_module('ltr')
    sys.modules['dlframework'] = sys.modules['ltr']
    sys.modules['dlframework.common'] = sys.modules['ltr']
    for m in ('model_constructor', 'stats', 'settings', 'local'):
        importlib.import_module('ltr.admin.'+m)
        sys.modules['dlframework.common.utils.'+m] = sys.modules['ltr.admin.'+m]


def _cleanup_legacy_env():
    del_modules = []
    for m in sys.modules.keys():
        if m.startswith('dlframework'):
            del_modules.append(m)
    for m in del_modules:
        del sys.modules[m]
