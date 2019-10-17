
//
//  hls_cnn.cpp
//  SqueezeNet
//
//  Created by Thomas Hou on 2019/10/14.
//  Copyright © 2019 Thomas Hou. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#if 0
#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_uint<2> u2;
typedef ap_uint<4> u4;
typedef ap_uint<8> u8;
typedef ap_uint<10> u10;
typedef ap_uint<12> u12;
typedef ap_uint<13> u13;
typedef ap_uint<16> u16;
typedef ap_uint<19> u19;
typedef ap_int<8> s8;
#else
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef uint8_t u2;
typedef uint8_t u4;
typedef uint8_t u8;
typedef uint16_t u10;
typedef uint16_t u12;
typedef uint16_t u13;
typedef uint16_t u16;
typedef uint32_t u19;
typedef char s8;
#endif

#define k3s2p0      1
#define k1s1p0      2
#define k3s1p1      4
#define avg         16
#define S_Float(f) (*((int*)(&f)) >>31)

void add(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
         u10 ch, u10 co, u8 ww, u8 wo, u2 k, u2 s, u2 p, u8 type);

void convolution(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
                 u10 ch, u10 co, u8 ww, u8 wo, u2 k, u2 s, u2 p);

