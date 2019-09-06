#!/bin/sh

#  build.sh
#  SqueezeNet
#
#  Created by Thomas Hou on 2019/9/6.
#  Copyright © 2019 Thomas Hou. All rights reserved.

gcc main.c -lm -lnnpack -lcpuinfo -lpthread -lpthreadpool -lclog -Os
