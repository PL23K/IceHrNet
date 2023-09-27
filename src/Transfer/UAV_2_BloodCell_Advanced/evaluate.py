# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import evaluate
from paddleseg.utils import logger, utils

import paddleseg.transforms as T
from paddleseg.datasets import Dataset


def main():
    cfg = Config('./icehrnet_icehrnetw48_40k_eval.yml')
    builder = SegBuilder(cfg)
    device_target = 'gpu:0'
    model_path = './output/20230915070734/best_model/model.pdparams'
    num_workers = 2

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(device)

    model = builder.model
    utils.load_entire_model(model, model_path)
    logger.info('Loaded trained weights successfully.')

    transforms = [
        T.Normalize()
    ]

    test_dataset = Dataset(
        dataset_root='../../../dataset/Transfer/UAV_2_BloodCell/test',
        transforms=transforms,
        val_path='../../../dataset/Transfer/UAV_2_BloodCell/test/test.txt',
        num_classes=2,
        mode='val'
    )

    evaluate(model, test_dataset, num_workers=num_workers)


if __name__ == '__main__':
    main()

# No resize
# 2023-09-15 19:36:41 [INFO]	[EVAL] #Images: 80 mIoU: 0.8704 Acc: 0.9342 Kappa: 0.8610 Dice: 0.9305
# 2023-09-15 19:36:41 [INFO]	[EVAL] Class IoU:
# [0.8987 0.8421]
# 2023-09-15 19:36:41 [INFO]	[EVAL] Class Precision:
# [0.9646 0.8877]
# 2023-09-15 19:36:41 [INFO]	[EVAL] Class Recall:
# [0.9293 0.9425]