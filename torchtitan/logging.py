# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch


class DistributedLogger(logging.getLoggerClass()):
    def info(self, msg, *args, **kwargs):
        # Check if this is the master rank
        if torch.dist.is_initialized() and torch.dist.get_rank() != 0:
            return  # Skip logging if not master rank
        super().info(msg, *args, **kwargs)


logging.setLoggerClass(DistributedLogger)  # Set the custom logger
logger = logging.getLogger()


def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"
