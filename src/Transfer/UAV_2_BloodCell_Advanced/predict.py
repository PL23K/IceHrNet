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
from paddleseg.utils import logger, get_image_list, utils
from paddleseg.core import predict
from paddleseg.transforms import Compose


def main():
    cfg = Config('./icehrnet_icehrnetw48_40k_eval.yml')
    builder = SegBuilder(cfg)
    device_target = 'gpu:0'
    model_path = './output/20230915070734/best_model/model.pdparams'

    image_path = '../../../dataset/Transfer/UAV_2_BloodCell/test/test.txt'
    save_dir = './output/20230915070734/best_model/test_result'

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(device_target)

    model = builder.model
    transforms = Compose(builder.val_transforms)
    image_list, image_dir = get_image_list(image_path)
    logger.info('The number of images: {}'.format(len(image_list)))

    predict(
        model,
        model_path=model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=save_dir)


if __name__ == '__main__':
    main()
