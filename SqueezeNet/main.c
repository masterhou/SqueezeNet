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

// Image memery [W * H * C] need convert to Data formate
// Data  [N * C * W * H]

typedef struct layer layer;
typedef void(*forword_func)(layer *l, uint8_t *d);

typedef struct layer{
    /* input */
    uint16_t c; //channel is input layer's filters
    uint16_t w;
    uint16_t h;
    /* filters / out channel */
    uint16_t co;
    /* convolution / pooling kernel */
    uint8_t k;
    uint8_t s; //stride
    uint8_t pad;
    /* layer type */
    uint32_t type; // bitwise
    /* data offset */
    uint32_t din;
    uint32_t dout;
    /* weight offset */
    uint32_t weight;
    uint32_t bias;
    layer* in;
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
    uint8_t channel;
    uint16_t input_w;
    uint16_t input_h;
    /* data offset */
    uint32_t image;
    uint8_t data[1024*1024*256]; // in DRAM
}network;

void build_layer(network* net, uint32_t type, uint8_t ci, uint8_t co, uint8_t k, uint8_t s, uint8_t pad, uint8_t in){
    layer *l = net->layers + net->llen;
    if(net->llen==0) {
        l->w = net->input_w;
        l->h = net->input_h;
        l->c = net->channel;
        l->din = net->image;
        l->dout = net->image + l->w*l->h*l->c*sizeof(float);
        l->weight = 0;
        l->bias = 0;
    }
    else {
        l->in = l-in;
        l->c = ci;
        l->co = co;
        l->w = l->in->w / l->in->s;
        l->h = l->in->h / l->in->s;
        
        l->din = l->in->dout;
        l->dout = l->din + l->w*l->h*l->c*sizeof(float);
        l->weight = l->in->weight + l->in->k * l->in->k * l->c * l->w * l->h * sizeof(float);
    }
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
        for (h=0; h<l->h; h+=l->s) {
            for (w=0; w<l->w; w+=l->s) { // stride
                float val = 0.f;
                for (c=0; c<l->c; c++) {
                    // convolution with kernel
                    for (kh=0; kh<l->k; kh++) {
                        for (kw=0; kw<l->k; kw++) {
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
        for (h=0; h<l->h; h+=l->s) {
            for (w=0; w<l->w; w+=l->s) { // stride
                float val = MIN_FLOAT;
                for (kh=0; kh<l->k; kh++) {
                    for (kw=0; kw<l->k; kw++) {
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
    uint16_t c,w,h,kw,kh;
    float *din=(float*)(d+l->din);
    float *out=(float*)(d+l->dout);
    for (c=0; c<l->c; c++) {
        for (h=0; h<l->h; h+=l->s) {
            for (w=0; w<l->w; w+=l->s) { // stride
                float val = MIN_FLOAT;
                for (kh=0; kh<l->k; kh++) {
                    for (kw=0; kw<l->k; kw++) {
                        val += din[c*(l->w*l->h) + l->w*(h+kh) + (w+kw)];
                    }
                }
                *(out++) = val/(l->s*l->s);
            }
        }
    }
}

void network_forword(network* net){
    int i;
    for (i=0; i<net->llen; i++) {
        net->layers[i].forword(&net->layers[i], net->data);
    }
}

void load_weight(network* net, const char* path){
    FILE   *f;
    f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long weight = ftell(f);
    fseek(f, 0, SEEK_CUR);
    fread(net->data, 1, weight, f);
    fclose(f);
    net->image = (uint32_t)weight;
}

void load_image(network* net, const char* path){
    int w, h, c;
    unsigned char *img = stbi_load(path, &w, &h, &c, 0);
    float *dat = (float*)(net->data + net->image);
    uint8_t *temp = net->data + net->image + net->input_w*net->input_h*net->channel*sizeof(float);
    // resize
    stbir_resize_uint8(img, w, h, 0, temp, net->input_w, net->input_h, 0, net->channel);
    // convert to C * W * H, float
    for (c=0; c<net->channel; c++) {
        for (h=0; h<net->input_h; h++) {
            for (w=0; w<net->input_w; w++) {
                *dat++ = (float)temp[net->input_w*h + w + c];
            }
        }
    }
    stbi_image_free(img);
}

void load_label(network* net, const char* path){
}

void build_fire(network* net, uint16_t inplanes, uint16_t squeeze_planes, uint16_t expand1x1_planes, uint16_t expand3x3_planes){
    // squeeze
    build_layer(net, L_CONV|L_RELU, inplanes, squeeze_planes, 1, 1, 0, 1);
    // squeeze -> expand1x1
    build_layer(net, L_CONV|L_RELU, squeeze_planes, expand1x1_planes, 1, 1, 0, 1);
    // squeeze -> expand3x3
    build_layer(net, L_CONV|L_RELU, squeeze_planes, expand3x3_planes, 3, 1, 1, 2);
    // concat: expand1x1 + expand3x3
    // *may use memery composition instead of concat layer: channel data + channel data
    build_layer(net, L_CONCAT, 3, 64, 3, 2, 0, 1);
}

/* SqueezeNet v1.1 */
void build_SqueezeNet_v11(network* net){
    // relu_conv1
    build_layer(net, L_CONV|L_RELU, 3, 64, 3, 2, 0, 0);
    build_layer(net, L_MAX, 64, 64, 3, 2, 0, 1);
    
    build_fire(net,  64, 16, 64, 64);
    build_fire(net, 128, 16, 64, 64);
    build_layer(net, L_MAX, 3, 64, 3, 2, 0, 1);
    
    build_fire(net, 128, 32, 128, 128);
    build_fire(net, 256, 32, 128, 128);
    build_layer(net, L_MAX, 3, 64, 3, 2, 0, 1);
    
    build_fire(net, 256, 48, 192, 192);
    build_fire(net, 384, 48, 192, 192);
    build_fire(net, 384, 64, 256, 256);
    build_fire(net, 512, 64, 256, 256);

    // Final convolution
    build_layer(net, L_CONV|L_RELU, 64, net->classes, 1, 1, 0, 1);
    build_layer(net, L_AVG, 1, 16, 1, 1, 0, 1);
}

/* SqueezeNet v1.0 */
void build_SqueezeNet_v10(network* net){
}

int main(int argc, const char * argv[]) {
    network net_v11;
    net_v11.llen = 0;
    net_v11.channel = 3;
    net_v11.classes = 1000;
    net_v11.input_w = 227;
    net_v11.input_h = 227;
    
    load_weight(&net_v11, argv[1]);
    load_label(&net_v11, argv[2]);
    load_image(&net_v11, argv[3]); // raw image, [c * w * h]
    build_SqueezeNet_v11(&net_v11);
    network_forword(&net_v11);
    return 0;
}
