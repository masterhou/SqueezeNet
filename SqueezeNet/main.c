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
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MIN_FLOAT       -1e6

// layer type
#define L_CONV  1
#define L_RELU  2
#define L_MAX   4
#define L_AVG   8
#define L_CONCAT 16
#define L_FIRE  32

char label[1000][160];

// Image memery [W * H * C] need convert to Data formate
// Data  [N * C * W * H]

typedef struct layer layer;
typedef struct network network;
typedef void(*forword_func)(network* net, layer *l);

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
    /* weight length */
    uint32_t weight;
    layer* in;
    forword_func forword; // funcation
} layer;

void conv_layer(network* net, layer *l);
void max_layer (network* net, layer *l);
void avg_layer (network* net, layer *l);
void cat_layer(network* net, layer *l);

#define LAY_MAX     0x40
typedef struct network {
    layer layers[LAY_MAX];
    uint8_t llen;
    uint16_t classes;
    uint8_t input_c;
    uint16_t input_w;
    uint16_t input_h;
    /* data offset */
    uint32_t weight;
    // weight + [layer: in(image in 1st layer / last layer's out) + out(next layer's in)]
    // weight lenght: [cout * cin * kernel^2]
    // bias lenght: [cout]
    float data[1024*1024*40]; // in DRAM
    float *dw;
}network;

void print_layer(layer *l) {
    printf("%s, %3d, %3d, %4d, %4d, %d, %d, %d, %6d, %7d, %7d\n", "conv", l->w, l->h, l->co, l->c, l->k, l->s, l->pad, l->weight, l->din, l->dout);
}

void print_wetwork(network* net) {
    int i = 0;
    printf(" i, type, wid, hei, cout,  cin, k, s, p, weight, data in, data out\n");
    for (i=0; i<net->llen; i++) {
        printf("%2d: ", i);
        print_layer(net->layers + i);
    }
}

#define DL(l) (l->w * l->h * l->c)

layer* build_layer(network* net, uint32_t type, uint16_t ci, uint16_t co, uint8_t k, uint8_t s, uint8_t pad, uint8_t in){
    layer *l = net->layers + net->llen;
    l->type = type;
    if(net->llen==0) {
        l->w = net->input_w;
        l->h = net->input_h;
        l->c = net->input_c;
    }
    else {
        l->in = l-in;
        l->c = l->in->co;
        l->w = l->in->w / l->in->s;
        l->h = l->in->h / l->in->s;
        l->din = l->in->dout;
    }
    l->co = co;
    l->k = k;
    l->s = s;
    l->pad = pad;

    l->dout = l->din + DL(l);
    
    if((type&L_CONV) == L_CONV){
        l->forword = conv_layer;
        l->weight = l->co * l->k * l->k * l->c;
    }
    else if((type&L_MAX) == L_MAX){
        l->forword = max_layer;
    }
    else if((type&L_AVG) == L_AVG){
        l->forword = avg_layer;
    }
    // else if((type&L_CONCAT) == L_CONCAT) {
    //     l->forword = cat_layer;
    // }
    
    net->llen++;
    return l;
}

// image pixel channle, todo PAD
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
void conv_layer(network* net, layer *l) {
    uint8_t relu = ((l->type & L_RELU) == L_RELU);
    uint16_t c,co,w,h,kw,kh;
    float *din =(float*)(net->data + l->din);
    float *out =(float*)(net->data + l->dout);
    float *wei =(float*)(net->dw);
    float *bias=(float*)(net->dw+l->weight);
    for (co=0; co<l->co; co++) {
        for (h=0; h<l->h; h+=l->s) {
            for (w=0; w<l->w; w+=l->s) { // stride
                float val = 0.f;
                for (c=0; c<l->c; c++) {
                    // convolution with kernel
                    for (kh=0; kh<l->k; kh++) {
                        for (kw=0; kw<l->k; kw++) {
                            val += input(l, din, w+kw, h+kh, c) * wei[co*c*l->k*l->k + l->k*kh + kw];
                        }
                    }
                }
                val += bias[co];
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
    net->dw += l->weight + l->co; // weight point ++
    printf("%ld ", net->dw-net->data);
}

void cat_layer(network* net, layer *l) {
    // only channel + channel
    // change next layer din offset
}

void max_layer(network* net, layer *l) {
    uint16_t c,w,h,kw,kh;
    float *din =(float*)(net->data + l->din);
    float *out =(float*)(net->data + l->dout);
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

void avg_layer(network* net, layer *l) {
    uint16_t c,w,h,kw,kh;
    float *din =(float*)(net->data + l->din);
    float *out =(float*)(net->data + l->dout);
    for (c=0; c<l->c; c++) {
        for (h=0; h<l->h; h+=l->s) {
            for (w=0; w<l->w; w+=l->s) { // stride
                float val = 0;
                for (kh=0; kh<l->k; kh++) {
                    for (kw=0; kw<l->k; kw++) {
                        val += din[c*(l->w*l->h) + l->w*(h+kh) + (w+kw)];
                    }
                }
                *(out++) = val/(l->s*l->s);
            }
        }
    }
    out =(float*)(net->data + l->dout);
    for (int i = 0; i < 1000; ++i)
    {
        printf("%f,", out[i]);
    }
}

void network_forword(network* net){
    int i;
    net->dw = net->data;
    for (i=0; i<net->llen; i++) {
        printf("%d: ", i);
        net->layers[i].forword(net, net->layers+i);
        printf("\n");
    }
}

void load_weight(network* net, const char* path){
    FILE   *f;
    f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long weight = ftell(f);
    fseek(f, 0, SEEK_SET);
    fread(net->data, 1, weight, f);
    fclose(f);
    net->weight = weight>>2;
    // printf("%d,%d\n", net->weight, weight);
}

void load_image(network* net, const char* path){
    int w, h, c;
    unsigned char *img = stbi_load(path, &w, &h, &c, 0);
    float *dat = net->data + net->weight;
    float *temp = dat + net->input_w*net->input_h*net->input_c;
    // resize
    stbir_resize_uint8(img, w, h, 0, (uint8_t*)temp, net->input_w, net->input_h, 0, net->input_c);
//    stbi_write_bmp("cat_227.bmp", net->input_w, net->input_h, 3, temp); // test
    // convert to C * W * H, float
    for (c=0; c<net->input_c; c++) {
        for (h=0; h<net->input_h; h++) {
            for (w=0; w<net->input_w; w++) {
                *dat++ = (float)temp[net->input_w*h + w + c];
            }
        }
    }
    stbi_image_free(img);
}

void load_label(network* net, const char* path){
    int i;
    FILE   *f;
    f = fopen(path, "r");
    for (i=0; i<1000; i++) {
        char c, *line = label[i];
        int len = 0;
        while ( (c = fgetc(f) ) != EOF && c != '\n')
        {
            line[len++] = c;
            line[len] = '\0';
        }
    }
    fclose(f);
}
void build_cat(network* net, uint32_t type, uint16_t co, layer* l1, layer* l2){
    layer *l = net->layers + net->llen;
    l->w = l1->w;
    l->h = l1->h;
    l->s = 1;
    l->co = co;
    l->type = type;
    l->dout = l1->dout;
    l->forword = cat_layer;
    net->llen++;
}

void build_fire(network* net, uint16_t inplanes, uint16_t squeeze_planes, uint16_t expand1x1_planes, uint16_t expand3x3_planes){
    // squeeze
    build_layer(net, L_CONV|L_RELU, inplanes, squeeze_planes, 1, 1, 0, 1);
    // squeeze -> expand1x1
    layer* l1 = build_layer(net, L_CONV|L_RELU, squeeze_planes, expand1x1_planes, 1, 1, 0, 1);
    // squeeze -> expand3x3
    layer* l3 = build_layer(net, L_CONV|L_RELU, squeeze_planes, expand3x3_planes, 3, 1, 1, 2);
    l3->dout = l1->dout + DL(l1);
    // concat: expand1x1 + expand3x3
    // *may use memery composition instead of concat layer: channel data + channel data
    build_cat(net, L_CONCAT, expand1x1_planes+expand3x3_planes, l1, l3);
}

/* SqueezeNet v1.1 */
void build_SqueezeNet_v11(network* net){
    // relu_conv1
    build_layer(net, L_CONV|L_RELU, 3, 64, 3, 2, 0, 0);
    build_layer(net, L_MAX, 64, 64, 3, 2, 0, 1);
    
    build_fire(net,  64, 16, 64, 64);
    build_fire(net, 128, 16, 64, 64);
    build_layer(net, L_MAX, 128, 128, 3, 2, 0, 1);
    
    build_fire(net, 128, 32, 128, 128);
    build_fire(net, 256, 32, 128, 128);
    build_layer(net, L_MAX, 256, 256, 3, 2, 0, 1);
    
    build_fire(net, 256, 48, 192, 192);
    build_fire(net, 384, 48, 192, 192);
    build_fire(net, 384, 64, 256, 256);
    build_fire(net, 512, 64, 256, 256);

    // Final convolution
    build_layer(net, L_CONV|L_RELU, 512, net->classes, 1, 1, 0, 1);
    build_layer(net, L_AVG, 16, 1, 14, 1, 0, 1);
}

/* SqueezeNet v1.0 */
void build_SqueezeNet_v10(network* net){
}

network net_v11;
int main(int argc, const char * argv[]) {
    net_v11.llen = 0;
    net_v11.classes = 1000;
    net_v11.input_c = 3;
    net_v11.input_w = 227;
    net_v11.input_h = 227;

    load_weight(&net_v11, argv[1]);
    load_label(&net_v11, argv[2]);
    load_image(&net_v11, argv[3]); // raw image, [c * w * h]
    build_SqueezeNet_v11(&net_v11);
    print_wetwork(&net_v11);
    network_forword(&net_v11);
    return 0;
}
