# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""convert pytorch pth to mindspore ckpt."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch

from mindspore.common import dtype as mstype
from mindspore.common import Parameter, Tensor
from mindspore.train import load_checkpoint, save_checkpoint


def torch_load(pretrained_file):
    model = torch.load(pretrained_file, map_location="cpu")['model']
    prefix = "model."
    model_weights = []
    for key in model.keys():
        # print(key, flush=True)
        if "norm" in key:
            if "weight" in key:
                name = prefix + key.replace(".weight", ".gamma")
            elif "bias" in key:
                name = prefix + key.replace(".bias", ".beta")
            else:
                raise ValueError("Not Supported Norm!")
        elif "qkv" in key:
            name = prefix + key
        elif "head" in key:
            name = prefix + key
        else:
            name = prefix + key
        model_weights.append(name)

    keys = sorted(model_weights)
    print("{}".format("\n".join(keys)), flush=True)


def ms_load(pretrained_file):
    param_dict = load_checkpoint(pretrained_file)
    model_weights = []
    other_weights = []
    for key, value in param_dict.copy().items():
        if key.startswith("model"):
            if "k_bias" in key:
                other_weights.append(key)
            else:
                model_weights.append(key)
        else:
            other_weights.append(key)

    keys = sorted(model_weights)
    # print("====== model weights ======\n{}".format("\n".join(model_weights)), flush=True)
    # print("====== other weights ======\n{}".format("\n".join(other_weights)), flush=True)
    print("{}".format("\n".join(keys)), flush=True)


def conv_pth2ckpt(pth_file, ckpt_file):
    print("====== load pth file: {} ======".format(pth_file), flush=True)

    model = torch.load(pth_file, map_location="cpu")['model']
    prefix = "model."
    model_weights = []

    for key in model.keys():
        key_weight_dict = {}
        if "norm" in key:
            if "weight" in key:
                name = prefix + key.replace(".weight", ".gamma")
            elif "bias" in key:
                name = prefix + key.replace(".bias", ".beta")
            else:
                raise ValueError("Not Supported Norm!")
            key_weight_dict["name"] = name
            key_weight_dict["data"] = Parameter(
                Tensor(model[key].numpy(), dtype=mstype.float32), requires_grad=True)
        elif "qkv" in key:
            name = prefix + key
            key_weight_dict["name"] = name
            key_weight_dict["data"] = Parameter(
                Tensor(model[key].numpy(), dtype=mstype.float32), requires_grad=True)
        elif "head" in key:
            continue
        else:
            name = prefix + key
            if "relative_position_index" in key:
                key_weight_dict["name"] = name
                key_weight_dict["data"] = Parameter(
                    Tensor(model[key].numpy(), dtype=mstype.int32), requires_grad=False)
            else:
                key_weight_dict["name"] = name
                key_weight_dict["data"] = Parameter(
                    Tensor(model[key].numpy(), dtype=mstype.float32), requires_grad=True)

        model_weights.append(key_weight_dict)

    print("====== save ckpt file: {} ======".format(ckpt_file), flush=True)
    save_checkpoint(model_weights, ckpt_file)


def demo():
    # ms_pretrained_file = "/Users/kaierlong/Downloads/Ascend/Beit/beit_base_patch16_224_pt22k_ft22k_0000-1_490.ckpt"
    # ms_load(pretrained_file=ms_pretrained_file)

    ms_pretrained_file = "/Users/kaierlong/Downloads/Ascend/Beit/pretrained_beit.ckpt"
    ms_load(pretrained_file=ms_pretrained_file)

    # torch_pretrained_file = "/Users/kaierlong/Downloads/Ascend/Beit/beit_base_patch16_224_pt22k_ft22k.pth"
    # torch_load(pretrained_file=torch_pretrained_file)


def main():
    parser = argparse.ArgumentParser(description="convert pth file to ckpt file.")
    parser.add_argument("--pth_file", type=str, required=True, help="pretrained pth file from torch.")
    parser.add_argument("--ckpt_file", type=str, required=True, help="pretrained ckpt file for mindspore.")

    args = parser.parse_args()

    conv_pth2ckpt(
        pth_file=args.pth_file, ckpt_file=args.ckpt_file)


if __name__ == "__main__":
    demo()
    # main()
