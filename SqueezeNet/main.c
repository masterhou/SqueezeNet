//
//  main.c
//  SqueezeNet
//
//  Created by Thomas Hou on 2019/8/29.
//  Copyright © 2019 Thomas Hou. All rights reserved.
//

#include <stdio.h>
#include <float.h>
#define NNPACK
#ifdef NNPACK
#include <nnpack.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STBIR_DEFAULT_FILTER_DOWNSAMPLE  STBIR_FILTER_CATMULLROM
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// layer type
#define L_CONV  1
#define L_RELU  2
#define L_MAX   4
#define L_AVG   8
#define L_CONCAT 16
#define L_FIRE  32
#define L_SOFT  64

char label[1000][150];

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
    uint16_t wo;
    uint16_t ho;
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
    char name[16];
} layer;

void conv_layer(network* net, layer *l);
void max_layer (network* net, layer *l);
void gavg_layer (network* net, layer *l);
void cat_layer(network* net, layer *l);
void softmax_layer(network* net, layer *l);

void conv_nnp(network* net, layer *l);


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
    float data[1024*1024*5]; // in DRAM
    float *dw;
}network;

void print_layer(layer *l) {
    printf("%6s, %3d, %3d, %4d, %4d, %3d, %3d, %2d, %2d, %d, %6d, %7d, %7d, %7d\n", l->name,
           l->w, l->h, l->c, l->co, l->wo, l->ho, l->k, l->s, l->pad,
           l->weight, l->din, l->dout, l->co * (l->wo) * (l->ho));
}

void print_wetwork(network* net) {
    int i = 0;
    printf(" i,   type, win, hin,  cin, cout,wout,hout,  k,  s, p, weight, data in, dataout, data len\n");
    for (i=0; i<net->llen; i++) {
        printf("%2d: ", i);
        print_layer(net->layers + i);
    }
}

float * test_load(char* npy, long s) {
    FILE   *f;
    f = fopen(npy, "rb");
    fseek(f, 128, SEEK_SET);
    float *dat = malloc(sizeof(float)*s);
    long ll = fread(dat, sizeof(float), s, f);
    assert(ll==s);
    fclose(f);
    return dat;
}

void test_data(char* npy, network* net, layer *l) {
    long s = l->co*(l->ho)*(l->wo);
    float *dat = test_load(npy, s), *dd = dat;
    float *out =(float*)(net->data + l->dout);
    int diff = 0;
    for (int i=0; i<s; i++) {
        if(fabsf(((*out) - (*dd))/(*out))>0.000001)
        {
            printf("%d: %f, %f\n", i, *out++, *dd++);
            diff = 1;
        }
    }
    free(dat);
    printf("test out %s(%ld) %s\n", l->name, s, diff? "failed" : "ok");
}
void test_weight(char* npy, network* net, layer *l) {
    long s = l->co*l->c*(l->k)*(l->k);
    float *dat = test_load(npy, s), *dd = dat;
    float *wei =(float*)(net->dw);
    int diff = 0;
    for (int i=0; i<s; i++) {
        if(fabsf(((*wei) - (*dd))/(*wei))>0.000001)
        {
            printf("%d: %f, %f\n", i, *wei++, *dd++);
            diff = 1;
        }
    }
    free(dat);
    printf("test weight %s(%ld) %s\n\n", l->name, s, diff? "failed" : "ok");
}

layer* build_layer(network* net, uint32_t type, uint16_t co, uint8_t k, uint8_t s, uint8_t pad, uint8_t in){
    layer *l = net->layers + net->llen;
    l->type = type;
    if(net->llen==0) {
        l->w = net->input_w;
        l->h = net->input_h;
        l->c = net->input_c;
        l->din = net->weight;
    }
    else {
        l->in = l-in;
        l->c = l->in->co;
        l->w = l->in->wo;
        l->h = l->in->ho;
        l->din = l->in->dout;
    }
    l->co = co;
    l->k = k;
    l->s = s;
    l->pad = pad;

    l->dout = l->din + (l->w * l->h * l->c);
    
    if((type&L_CONV) == L_CONV){
        l->ho = (l->h + 2 * pad - k) / s + 1;
        l->wo = (l->w + 2 * pad - k) / s + 1;
        l->weight = l->co * l->c * l->k * l->k;
#ifdef NNPACK
        l->forword = conv_nnp;
#else
        l->forword = conv_layer;
#endif
        sprintf(l->name, "conv%d", net->llen);
    }
    else if((type&L_MAX) == L_MAX){
        l->ho = ceil((l->h + 2.0 * pad - k) / s) + 1;
        l->wo = ceil((l->w + 2.0 * pad - k) / s) + 1;
        l->forword = max_layer;
        sprintf(l->name, "max%d", net->llen);
    }
    else if((type&L_AVG) == L_AVG){
        l->ho = 1;
        l->wo = 1;
        l->forword = gavg_layer;
        sprintf(l->name, "gavg%d", net->llen);
    }
     else if((type&L_SOFT) == L_SOFT) {
         l->ho = 1;
         l->wo = 1;
         l->forword = softmax_layer;
         sprintf(l->name, "soft%d", net->llen);
     }
    
    net->llen++;
    return l;
}

// get input data
float input(layer *l, float *d, uint16_t w, uint16_t h, uint16_t c){
    // padding, fill 0.0
    
    w -= l->pad;
    h -= l->pad;
    if(w >= l->w || w < 0)
        return 0.0;
    if(h >= l->h || h < 0)
        return 0.0;
    
    return d[c*(l->w*l->h) + l->w*h + w];
}
#ifdef NNPACK
void conv_nnp(network* net, layer *l) {
    uint8_t relu = ((l->type & L_RELU) == L_RELU);
//    uint16_t c,co,w,h,kw,kh, tk=l->k*l->k;
    float *din =(float*)(net->data + l->din);
    float *out =(float*)(net->data + l->dout);
    float *wei =(float*)(net->dw);
    float *bias=(float*)(net->dw+l->weight);
    struct nnp_size in_size = {l->w, l->h};
    struct nnp_size st_size = {l->s, l->s};
    struct nnp_size kr_size = {l->k, l->k};
    struct nnp_padding pad_size = {l->pad,l->pad,l->pad,l->pad};
    int status = nnp_convolution_inference(nnp_convolution_algorithm_auto,
                              nnp_convolution_transform_strategy_tuple_based,
                              l->c, l->co, in_size, pad_size, kr_size, st_size,
                              din, wei, bias, out,
                              NULL, NULL, relu, NULL, NULL, NULL);
    
    if(status) printf("\n%s nnp status:%d\n", l->name, status);
    net->dw += l->weight + l->co; // weight point ++
}
#endif

// 2d image convolution
void conv_layer(network* net, layer *l) {
    uint8_t relu = ((l->type & L_RELU) == L_RELU);
    uint16_t c,co,w,h,kw,kh, tk=l->k*l->k;
    float *din =(float*)(net->data + l->din);
    float *out =(float*)(net->data + l->dout);
    float *wei =(float*)(net->dw);
    float *bias=(float*)(net->dw+l->weight);
    for (co=0; co<l->co; co++) {
        for (h=0; h<l->ho; h++) {
            for (w=0; w<l->wo; w++) { // stride
                float val = bias[co]; // add bias
                for (c=0; c<l->c; c++) {
                    // convolution with kernel
                    for (kh=0; kh<l->k; kh++) {
                        for (kw=0; kw<l->k; kw++) {
                            float weight = wei[co*l->c*tk + c*tk + l->k*kh + kw];
                            float p = input(l, din, l->s*w+kw, l->s*h+kh, c);
                            val +=  p * weight;
                        }
                    }
                    // use Accelerate
                    // cblas_sgemm(
                }
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
        for (h=0; h<l->ho; h++) {
            for (w=0; w<l->wo; w++) { // stride
                float val = FLT_MIN;
                for (kh=0; kh<l->k; kh++) {
                    for (kw=0; kw<l->k; kw++) {
                        float t = input(l, din, l->s*w+kw, l->s*h+kh, c);
                        if(t>val)
                            val = t;
                    }
                }
                *out++ = val;
            }
        }
    }
}

void gavg_layer(network* net, layer *l) {
    uint32_t c,i,size = l->h*l->w;
    float *din =(float*)(net->data + l->din);
    float *out =(float*)(net->data + l->dout);
    for (c=0; c<l->c; c++) {
        float sum = 0.f;
        for (i=0; i<size; i++) {
            sum += *din++;
        }
        *out++ = sum/size;
//        *(out++) = c; // for sort and index
    }
//    test_data("pool10.npy", net, l);
}

void avg_layer(network* net, layer *l) {
    uint16_t c,w,h,kw,kh;
    float *din =(float*)(net->data + l->din);
    float *out =(float*)(net->data + l->dout);
    for (c=0; c<l->c; c++) {
        for (h=0; h<l->ho; h++) {
            for (w=0; w<l->wo; w++) { // stride
                float val = 0;
                for (kh=0; kh<l->k; kh++) {
                    for (kw=0; kw<l->k; kw++) {
                        val += input(l, din, l->s*w+kw, l->s*h+kh, c);
                    }
                }
                *out++ = val/(l->k*l->k);
            }
        }
    }
}

void softmax_layer(network* net, layer *l) {
    float *in  =(float*)(net->data + l->din);
    float *out =(float*)(net->data + l->dout);
    float sum=0;
    for (int i=0; i<net->classes; i++) {
        *out = expf(*in++);
        sum += *out++;
        *out++ = i; // for sort and index
    }
    
    out =(float*)(net->data + l->dout);
    for (int i=0; i<net->classes; i++) {
        *out = *out / sum;
        out++;
        *out++ = i; // for sort and index
    }

}

int compar(const void* a, const void* b) {
    return (*(float*)b - *(float*)a)*10000;
}

void prob(network* net, layer *l) {
    float *out =(float*)(net->data + l->dout);
    qsort(out, net->classes, 8, compar);
    for (int i=0; i<5; ++i) {
        float val = out[i*2];
        int index = out[i*2+1];
        printf("classify result [%d]: %f, %s\n", index, val, label[index]);
    }
}

void network_forword(network* net){
    int i;
    net->dw = net->data;
    for (i=0; i<net->llen; i++) {
//        printf("%d: %s", i, net->layers[i].name);
        net->layers[i].forword(net, net->layers+i);
//        printf("\n");
//        if(i==0){ // ok
//            test_data("conv1.npy", net, net->layers+i);
//            break;
//        }
//        if(i==10){ // ok
//            test_data("pool3.npy", net, net->layers+i);
//            break;
//        }
//        if(i==19){  // ok
//            test_data("pool5.npy", net, net->layers+i);
//            break;
//        }
//        if(i==23){  // fcat23 / fire6/concat ok
//            test_data("fire6_concat.npy", net, net->layers+i);
//            break;
//        }
//        if(i==24){  // fire27 ok
//            test_data("fire7_squeeze1x1.npy", net, net->layers+i);
//            break;
//        }
//        if(i==25){  // ok
//            test_data("fire7_expand1x1.npy", net, net->layers+i);
//            break;
//        }
//        if(i==26){  // ok
//            test_data("fire7_expand3x3.npy", net, net->layers+i);
//            break;
//        }
    }
    prob(net, net->layers + net->llen-1);
    return;
}

layer* build_cat(network* net, uint32_t type, layer* l1, layer* l2){
    layer *l = net->layers + net->llen;
    l->c = l1->c;
    l->w = l1->w;
    l->h = l1->h;
    l->wo = l1->wo;
    l->ho = l1->ho;
    l->s = 1;
    l->co = l1->co + l2->co;
    l->type = type;
    l->din = l1->din;
    l->dout = l1->dout;
    l->forword = cat_layer;
    net->llen++;
    return l;
}

void build_fire(network* net, uint16_t inplanes, uint16_t squeeze_planes, uint16_t expand1x1_planes, uint16_t expand3x3_planes){
    // squeeze
    static int i=2;
    layer* l = build_layer(net, L_CONV|L_RELU, squeeze_planes, 1, 1, 0, 1);
    sprintf(l ->name, "f%d/s1", i);
    // squeeze -> expand1x1
    layer* l1 = build_layer(net, L_CONV|L_RELU, expand1x1_planes, 1, 1, 0, 1);
    sprintf(l1->name, "f%d/e1", i);
    // squeeze -> expand3x3
    layer* l3 = build_layer(net, L_CONV|L_RELU, expand3x3_planes, 3, 1, 1, 2);
    l3->dout = l1->dout + l1->co * (l1->wo) * (l1->ho);
    sprintf(l3->name, "f%d/e3", i);
    // concat: expand1x1 + expand3x3
    // *may use memery composition instead of concat layer: channel data + channel data
    l = build_cat(net, L_CONCAT, l1, l3);
    sprintf(l->name, "f%d/cat", i);
    i++;
}

/* SqueezeNet v1.1 */
void build_SqueezeNet_v11(network* net){
    // relu_conv1
    build_layer(net, L_CONV|L_RELU, 64, 3, 2, 0, 0);
    build_layer(net, L_MAX, 64, 3, 2, 0, 1);
    
    build_fire(net,  64, 16, 64, 64);
    build_fire(net, 128, 16, 64, 64);
    build_layer(net, L_MAX, 128, 3, 2, 0, 1);
    
    build_fire(net, 128, 32, 128, 128);
    build_fire(net, 256, 32, 128, 128);
    build_layer(net, L_MAX, 256, 3, 2, 0, 1);
    
    build_fire(net, 256, 48, 192, 192);
    build_fire(net, 384, 48, 192, 192);
    build_fire(net, 384, 64, 256, 256);
    build_fire(net, 512, 64, 256, 256);

    // Final convolution
    layer *l=build_layer(net, L_CONV|L_RELU, net->classes, 1, 1, 0, 1);
    build_layer(net, L_AVG, net->classes, l->wo, l->ho, 0, 1);
    build_layer(net, L_SOFT, net->classes, 1, 1, 0, 1);
}

/* SqueezeNet v1.0 */
void build_SqueezeNet_v10(network* net){
}

void load_weight(network* net, const char* path){
    FILE   *f;
    f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long weight = ftell(f);
    fseek(f, 0, SEEK_SET);
    fread(net->data, 1, weight, f);
    fclose(f);
    net->weight = (int32_t)weight>>2;
}

void load_image(network* net, const char* path){
    int w, h, c;
    uint8_t *img = stbi_load(path, &w, &h, &c, 0);
    float *dat = net->data + net->weight;
    uint8_t temp[w*h*c];
    // resize
    stbir_resize_uint8(img, w, h, 0, temp, net->input_w, net->input_h, 0, 3);
    // stbi_write_bmp("cat_227.bmp", net->input_w, net->input_h, 3, temp); // test
    // convert to C * W * H, float
    // mean-subtracted values:
    float BGR[] = {104.00698793, 116.66876762, 122.67891434};
    for (c=0; c<net->input_c; c++) {
        for (h=0; h<net->input_h; h++) {
            for (w=0; w<net->input_w; w++) {
                *dat++ = temp[net->input_w*3*h + w*3 + (2-c)] - BGR[c];
            }
        }
    }
    // dat = net->data + net->weight;
    // for (c=0; c<net->input_c; c++) {
    //     for (h=0; h<net->input_h; h++) {
    //         for (w=0; w<net->input_w; w++) {
    //                printf("%f, ", *dat++);
    //             }
    //         }
    // }
}

void load_img_npy(network* net, const char* path){
    FILE   *f;
    f = fopen(path, "rb");
    float *dat = net->data + net->weight;
    fseek(f, 128, SEEK_SET);
    long s = net->input_c * net->input_w * net->input_h;
    long l = fread(dat, sizeof(float), s, f);
    assert(l==s);
    fclose(f);
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

network net_v11;
int main(int argc, const char * argv[]) {
#ifdef NNPACK
    nnp_initialize();
#endif
    net_v11.llen = 0;
    net_v11.classes = 1000;
    net_v11.input_c = 3;
    net_v11.input_w = 227;
    net_v11.input_h = 227;

    load_weight(&net_v11, argv[1]);
    load_label(&net_v11, argv[2]);
    load_image(&net_v11, argv[3]); // raw image, [c * w * h]
//     load_img_npy(&net_v11, argv[3]);
    build_SqueezeNet_v11(&net_v11);
#ifdef DEBUG
    print_wetwork(&net_v11);
#endif
    network_forword(&net_v11);
    
    printf("\n");
    return 0;
}
