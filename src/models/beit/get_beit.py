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
"""Get Beit of different size for args"""
from .beit import beit_base_patch16_224


def beit_base_patch16_224_pt22k_ft22k(args):
    """get beit_base_patch16_224_pt22k_ft22k according to args"""
    # override args
    image_size = args.image_size
    num_classes = args.num_classes
    drop_path_rate = args.drop_path_rate
    use_rel_pos_bias = args.rel_pos_bias
    use_abs_pos_emb = args.abs_pos_emb
    init_values = args.layer_scale_init_value

    print(25 * "=" + "MODEL CONFIG" + 25 * "=")
    print(f"==> IMAGE_SIZE:         {image_size}")
    print(f"==> NUM_CLASSES:        {num_classes}")
    print(f"==> DROP_PATH_RATE:     {drop_path_rate}")
    print(f"==> USE_REL_POS_BIAS:                {use_rel_pos_bias}")
    print(f"==> USE_ABS_POS_EMB:         {use_abs_pos_emb}")
    print(f"==> INIT_VALUES:         {init_values}")
    print(25 * "=" + "FINISHED" + 25 * "=")

    model = beit_base_patch16_224(
        drop_path_rate=args.drop_path_rate,
        num_classes=args.num_classes,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value)

    # print(model, flush=True)
    return model
