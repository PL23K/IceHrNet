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
from paddleseg.utils import utils
from paddleseg.core import train
import datetime

def main():
    cfg = Config('./icehrnet_icehrnetbackbonew48__40k.yml')
    builder = SegBuilder(cfg)
    device = 'gpu'
    device_target = 'gpu:0'
    save_dir = './output/'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    resume_model = None
    save_interval = 500
    log_iters = 10
    num_workers = 0
    use_vdl = True
    use_ema = False
    keep_checkpoint_max = 1  # Maximum number of checkpoints to save.
    # Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal.
    precision = 'fp32'
    amp_level = '01'
    profiler_options = None

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(None)
    utils.set_device(device_target)
    utils.set_cv2_num_threads(num_workers)

    model = utils.convert_sync_batchnorm(builder.model, device)

    train_dataset = builder.train_dataset
    val_dataset = builder.val_dataset
    optimizer = builder.optimizer
    loss = builder.loss

    train(
        model,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        save_dir=save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=resume_model,
        save_interval=save_interval,
        log_iters=log_iters,
        num_workers=num_workers,
        use_vdl=use_vdl,
        use_ema=use_ema,
        losses=loss,
        keep_checkpoint_max=keep_checkpoint_max,
        test_config=cfg.test_config,
        precision=precision,
        amp_level=amp_level,
        profiler_options=profiler_options,
        to_static_training=cfg.to_static_training)


if __name__ == '__main__':
    main()

# 351/351 [==============================] - 59s 169ms/step - batch_cost: 0.1693 - reader cost: 1.7087e-04
# 2023-09-15 05:04:49 [INFO]      [EVAL] #Images: 351 mIoU: 0.9596 Acc: 0.9829 Kappa: 0.9586 Dice: 0.9793
# 2023-09-15 05:04:49 [INFO]      [EVAL] Class IoU: 
# [0.9761 0.9431]
# 2023-09-15 05:04:49 [INFO]      [EVAL] Class Precision: 
# [0.9963 0.9512]
# 2023-09-15 05:04:49 [INFO]      [EVAL] Class Recall: 
# [0.9796 0.9911]
# 2023-09-15 05:04:50 [INFO]      [EVAL] The model with the best validation mIoU (0.9734) was saved at iter 12000.
# <class 'paddle.nn.layer.conv.Conv2D'>'s flops has been counted
# <class 'paddle.nn.layer.norm.BatchNorm2D'>'s flops has been counted
# <class 'paddle.nn.layer.activation.ReLU'>'s flops has been counted
# Cannot find suitable count function for <class 'paddleseg.models.layers.wrap_functions.Add'>. Treat it as zero FLOPs.
# Cannot find suitable count function for <class 'paddle.nn.layer.common.Identity'>. Treat it as zero FLOPs.
# <class 'paddle.nn.layer.pooling.AdaptiveAvgPool2D'>'s flops has been counted
# <class 'paddle.nn.layer.common.Dropout'>'s flops has been counted
# Total Flops: 113124160720     Total Params: 66853746