//
//  main.c
//  SqueezeNet
//
//  Created by Thomas Hou on 2019/8/29.
//  Copyright © 2019 Thomas Hou. All rights reserved.
//

#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

#define MIN_FLOAT       -1e6

uint8_t data[1024*1024*256]; // in DRAM
uint32_t weight_size = 0;
uint32_t image_size = 0;

// Image memery [W * H * C] need convert to Data formate
// Data  [N * C * W * H]

typedef struct layer layer;
typedef void(*forword_func)(layer *l, uint8_t *d);

typedef struct layer{
    /* input */
    uint16_t c;
    uint16_t w;
    uint16_t h;
    /* output */
    uint16_t co;
    uint16_t wo;
    uint16_t ho;
    /* convolut or pool kernel */
    uint8_t k;
    uint8_t s; //stride
    uint8_t pad;
    /* layer type */
    uint32_t type; // bit
    /* data offset */
    uint32_t din;
    uint32_t dout;
    /* weight offset */
    uint32_t weight;
    uint32_t bias;

    forword_func forword; // funcation
} layer;
// layer type
#define L_CONV  1
#define L_RELU  2
#define L_MAX   4
#define L_AVG   8
#define L_CONCAT 16
#define L_FIRE  32

void conv_layer(layer *l, uint8_t *d);
void max_layer(layer *l, uint8_t *d);
void avg_layer(layer *l, uint8_t *d);
void concat_layer(layer *l, uint8_t *d);


#define LAY_MAX     0x10
typedef struct network {
    layer layers[LAY_MAX];
    uint8_t llen;
    uint16_t classes;
}network;

void build_layer(network* net, uint32_t type, uint16_t w, uint16_t h, uint8_t c,
                uint8_t co, uint8_t k, uint8_t s, uint8_t pad){
    layer *l = net->layers + net->llen;
    l->c = c;
    l->w = w;
    l->h = h;
    l->co = co;
    l->k = k;
    l->s = s;
    l->pad = pad;
    if((type&L_CONV) == L_CONV)
        l->forword = conv_layer;
    else if((type&L_MAX) == L_MAX)
        l->forword = max_layer;
    else if((type&L_AVG) == L_AVG)
        l->forword = avg_layer;
    else if((type&L_CONCAT) == L_CONCAT)
        l->forword = concat_layer;
    
    net->llen++;
}

// image pixel channle
float input(layer *l, float *d, uint16_t w, uint16_t h, uint8_t c){
    // ignore pad
    w -= l->k/2;
    h -= l->k/2;
    if(w > l->w)    w = l->w - 1;
    else if(w < 0)  w = 0;
    
    if(h > l->h)    h= l->h - 1;
    else if(h < 0)  h = 0;
    
    return d[c*(l->w*l->h) + l->w*h + w];
}

// 2d image convolution
void conv_layer(layer *l, uint8_t *d) {
    uint8_t relu = ((l->type & L_RELU) == L_RELU);
    uint16_t c,co,w,h,kw,kh;
    float *din=(float*)(d+l->din);
    float *out=(float*)(d+l->dout);
    float *wei=(float*)(d+l->weight);
    float *bias=(float*)(d+l->bias);
    for (co=0; co<l->co; co++) {
        for (w=0; w<l->w; w+=l->s) { // stride
            for (h=0; h<l->h; h+=l->s) {
                float val = 0.f;
                for (c=0; c<l->c; c++) {
                    // convolution with kernel
                    for (kw=0; kw<l->k; kw++) {
                        for (kh=0; kh<l->k; kh++) {
                            // val += input(l, din, w+kw, h+kh, c) * wei[co*c*l->k*l->k + l->k*kh + kw];
                            val += input(l, din, w+kw, h+kh, c) * (*wei++); // weight as stream instead of index
                        }
                    }
                }
                val += *bias++;
                // ReLU
                if(relu && val<=0.f){
                    // index = co * (l->w/s) * (l->h/s) + (l->w/s)*(h/s) + w/s;
                    *out++ = 0.f; // out as stream instead of index
                }
                else {
                    *out++ = val;
                }
            }
        }
    }
}

void concat_layer(layer *l, uint8_t *d) {
    
}

void max_layer(layer *l, uint8_t *d) {
    uint16_t c,w,h,kw,kh;
    float *din=(float*)(d+l->din);
    float *out=(float*)(d+l->dout);
    for (c=0; c<l->c; c++) {
        for (w=0; w<l->w; w+=l->s) { // stride
            for (h=0; h<l->h; h+=l->s) {
                float val = MIN_FLOAT;
                for (kw=0; kw<l->k; kw++) {
                    for (kh=0; kh<l->k; kh++) {
                        float t = din[c*(l->w*l->h) + l->w*(h+kh) + (w+kw)];
                        if(t>val)
                            val = t;
                    }
                }
                *(out++) = val;
            }
        }
    }
}

void avg_layer(layer *l, uint8_t *d) {
    uint16_t c,i;
    float *din=(float*)(d+l->din);
    float *out=(float*)(d+l->dout);
    for (c=0; c<l->c; c++) {
        uint32_t k = l->c;
        out[k] = 0;
        for (i=0; i<l->h*l->w; i++) {
            out[c] += din[i];
        }
        out[c] /= l->h*l->w;
    }
}

void network_forword(network* net, uint8_t *d){
    int i;
    for (i=0; i<net->llen; i++) {
        net->layers[i].forword(&net->layers[i], d);
    }
}

void load_weight(const char* path){
    FILE   *f;
    f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    weight_size = (uint32_t)ftell(f);
    fseek(f, 0, SEEK_CUR);
    fread(data, 1, weight_size, f);
    fclose(f);
}

void load_image(const char* path, int16_t w, int16_t h){
    int w_, h_, n;
    unsigned char *img = stbi_load(path, &w_, &h_, &n, 0);
    // resize and convert to C * W * H
    stbir_resize_uint8(img+0, w_, h_, 2, data+weight_size, w, h, 0, 1); //R channel
    stbir_resize_uint8(img+1, w_, h_, 2, data+w*h*1,       w, h, 0, 1); //G channel
    stbir_resize_uint8(img+2, w_, h_, 2, data+w*h*2,       w, h, 0, 1); //B channel
    stbi_image_free(img);
}

void load_label(const char* path){
}

void build_fire(network* net, uint16_t inplanes, uint16_t squeeze_planes, uint16_t expand1x1_planes, uint16_t expand3x3_planes){
    // squeeze
    build_layer(net, L_CONV|L_RELU, 227, 227, 3, 64, 3, 2, 0);
    // expand1x1
    build_layer(net, L_CONV|L_RELU, 227, 227, 3, 64, 3, 2, 0);
    // expand3x3
    build_layer(net, L_CONV|L_RELU, 227, 227, 3, 64, 3, 2, 0);
    // concat: expand1x1 + expand3x3
    // *may use memery composition instead of concat layer: channel data + channel data
    build_layer(net, L_CONCAT, 227, 227, 3, 64, 3, 2, 0);
}

/* SqueezeNet v1.1 */
void build_SqueezeNet_v11(network* net, uint16_t classes/*=1000*/){
    // relu_conv1
    build_layer(net, L_CONV|L_RELU, 227, 227, 3, 64, 3, 2, 0);
    build_layer(net, L_MAX, 113, 113, 3, 64, 3, 2, 0);
    
    build_fire(net,  64, 16, 64, 64);
    build_fire(net, 128, 16, 64, 64);
    build_layer(net, L_MAX, 113, 113, 3, 64, 3, 2, 0);
    
    build_fire(net, 128, 32, 128, 128);
    build_fire(net, 256, 32, 128, 128);
    build_layer(net, L_MAX, 113, 113, 3, 64, 3, 2, 0);
    
    build_fire(net, 256, 48, 192, 192);
    build_fire(net, 384, 48, 192, 192);
    build_fire(net, 384, 64, 256, 256);
    build_fire(net, 512, 64, 256, 256);

    // Final convolution
    build_layer(net, L_CONV|L_RELU, 512, 512, 64, classes, 1, 1, 0);
    build_layer(net, L_AVG, 512, 512, 1, 16, 1, 1, 0);
}

/* SqueezeNet v1.0 */
void build_SqueezeNet_v10(network* net){
}

int main(int argc, const char * argv[]) {
    network net_v11;
    net_v11.llen = 0;
    load_weight(argv[1]);
    load_image(argv[3], 227, 227); // raw image, [c * w * h]
    load_label(argv[2]);
    build_SqueezeNet_v11(&net_v11, 1000);
    network_forword(&net_v11, data);
    return 0;
}
