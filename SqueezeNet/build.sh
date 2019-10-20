#!/bin/sh

#  build.sh
#  SqueezeNet
#
#  Created by Thomas Hou on 2019/9/6.
#  Copyright © 2019 Thomas Hou. All rights reserved.

gcc hls_cnn.c synset_words.c main.c -lm -lnnpack -lcpuinfo -lpthread -lpthreadpool -lclog -Os
