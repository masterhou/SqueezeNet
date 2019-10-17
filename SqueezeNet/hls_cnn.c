//
//  hls_cnn.cpp
//  SqueezeNet
//
//  Created by Thomas Hou on 2019/10/14.
//  Copyright © 2019 Thomas Hou. All rights reserved.
//

#include "hls_cnn.h"

float sum9(float *buf){
#pragma HLS INLINE
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7] + buf[8];
}

float sum8(float *buf){
#pragma HLS INLINE
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
}

float sum_all(float *acc_t, float acc, u10 ch){
#pragma HLS INLINE

    add_acc_t:
    for(u10 j=0; j < ch; j+=16){
        acc += sum8(acc_t+j) + sum8(acc_t+j+8);
    }
    return acc;
}

float acc_t[128];
float out[56*128];
float wei[64*3*3];
float bb[1000];
float im[56*128]; // stride + 2*padding = 3
static u10 wei_len, wx3;
static u16 w2;

void conv_k3s1p1(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
u10 ch, u10 co, u8 ww, u8 wo) {
#pragma HLS INLINE
    u2 m, n;
    u8 x, y, j, mm;
    u10 i;
    u12 yj, y2;
    s8 xx, yy;
    u16 yw;
    float buf[9];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1

    volatile float *imp = img-ww; // for padding=1
    conv_k3s1p1:
    for(i=0; i < co; i++){
        memcpy(wei, (const void*)(weight), wei_len*sizeof(float));
        weight += wei_len;
        float bbi = bb[i];
        k3s1p1_y:
        for(y=0, y2=0; y<wo; y++, y2+=ww){
            for(j=0, yj=0, yw=0; j < ch; j++, yj+=wx3, yw+=w2){
                memcpy(im+yj, (void*)(imp+yw+y2), wx3*sizeof(float)); // k=3;s=1
            }
            k3s1p1_x:
            for(x=0; x<wo; x++){
                u16 wi = 0;
                k3s1p1_ch:
                for(j=0, yj=0; j < ch; j++, yj+=wx3){
                #pragma HLS PIPELINE II=1
                    u4 _i=0;
                    k3s1p1_kernel_y1:
                    for(m=0,mm=0,yy=y-1; m<3; m++,mm+=ww,yy++){
                        k3s1p1_kernel_x1:
                        for(n=0, xx=x-1; n<3; n++, xx++){
                            if(xx>=0 && xx<ww && yy>=0 && yy<ww){
                                buf[_i++] = wei[wi] * im[yj + mm + xx];
                            }
                            else {
                                buf[_i++] = 0.f;
                            }
                            wi++;
                        }
                    }
                    acc_t[j] = sum9(buf);
                }
                float acc = sum_all(acc_t, bbi, ch);
                *dout++ = S_Float(acc) ? 0.0f : acc;
            }
        }
    }
}


void conv_k3s2p0(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
u10 ch, u10 co, u8 ww, u8 wo){
#pragma HLS INLINE
    u2 m, n, j; // max = 3
    u8 x, y, xx, i;
    u10 mm;
    u12 yj;
    u16 y2;
    u19 yw;
//    float buf[9];
//#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
//    float acb[3];
//#pragma HLS ARRAY_PARTITION variable=acb complete dim=1

    conv_k3s2p0:
    for(i=0; i < 64; i++){
        float bbi = bb[i];
        memcpy(wei, (const void*)(weight), wei_len*sizeof(float));
        weight += wei_len;

        k3s2p0_y_out:
        for(y=0, y2=0; y<113; y++, y2+=ww*2){
            k3s2p0_img_cpy:
            for(j=0, yj=0, yw=0; j < 3; j++, yj+=wx3, yw+=w2){
#pragma HLS UNROLL
                memcpy(im+yj, (void*)(img+yw+y2), wx3*sizeof(float)); // k=3;s=2
            }
            k3s2p0_x_out:
            for(x=0; x<113; x++){
                u16 wi = 0;
                float acc = bbi;
                k3s2p0_ch:
                for(j=0, yj=0; j < 3; j++, yj+=wx3){
                #pragma HLS PIPELINE II=1
//                    u4 _i=0;
                    k3s2p0_y2:
                    for(m=0, mm=0; m<3; m++, mm+=ww){
                        k3s2p0_x2:
                        for(n=0, xx=(x<<1); n<3; n++, xx++){
//                            buf[_i++] = wei[wi++] * im[yj + mm + xx];
                            acc += wei[wi++] * im[yj + mm + xx];
                        }
                    }
//                    acb[j] = sum9(buf);
                }
//                float acc = bbi + acb[0] + acb[1] + acb[2];
                *dout++ = S_Float(acc) ? 0.0f : acc;
            }
        }
    }
}

void conv_k1s1p0(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
u10 ch, u10 co, u8 ww, u8 wo) {
#pragma HLS INLINE
    u8 x, y;
    u10 i, j;
    u12 yy;
    u13 yj;
    u19 yw;
    conv_k1s1p0:
    for(i=0; i < co; i++){
        memcpy(wei, (const void*)(weight), wei_len*sizeof(float));
        weight += wei_len;
        float bbi = bb[i];
        k1s1p0_y_out:
        for(y=0, yy=0; y<wo; y++, yy+=ww){
            for(j=0, yj=0, yw=0; j < ch; j++, yj+=ww, yw+=w2){
                memcpy(im+yj, (void*)(img+yw+yy), ww*sizeof(float));
            }
            k1s1p0_x_out:
            for(x=0; x<wo; x++){
                float acc = bbi;
                k1s1p0_k:
                for(j=0, yj=0; j < ch; j++, yj+=ww){
#pragma HLS UNROLL skip_exit_check factor=16
                    acc += wei[j] * im[yj + x];
                }
                *dout++ = S_Float(acc) ? 0.0f : acc;
            }
        }
    }
}

void avg_14_1000(volatile float *img, volatile float *dout){
    u10 j;
#pragma HLS INLINE
    avg_chan:
    for(j=0; j < 1000; j++){
#pragma HLS PIPELINE II=1
        memcpy(im, (void*)(img+ 14*14*j), 14*14*sizeof(float));

        float v = im[0]+im[1]+im[2]+im[3];
        v = sum_all(im+4, v, 16*12);

        dout[j] = v*0.005102040816327f;
    }
}

void add(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
    u10 ch, u10 co, u8 ww, u8 wo, u2 k, u2 s, u2 p, u8 type) {
#pragma HLS INTERFACE m_axi depth=1024 port=weight offset=slave
#pragma HLS INTERFACE m_axi depth=1024 port=img offset=slave
#pragma HLS INTERFACE m_axi depth=1024 port=bias offset=slave
#pragma HLS INTERFACE m_axi depth=1024 port=dout offset=slave
#pragma HLS INTERFACE s_axilite port=ch
#pragma HLS INTERFACE s_axilite port=co
#pragma HLS INTERFACE s_axilite port=ww
#pragma HLS INTERFACE s_axilite port=wo
#pragma HLS INTERFACE s_axilite port=k
#pragma HLS INTERFACE s_axilite port=s
#pragma HLS INTERFACE s_axilite port=p
#pragma HLS INTERFACE s_axilite port=type
#pragma HLS INTERFACE s_axilite port=return

    w2 = ww*ww;
    wx3 = ww*3;
    if(k==1)
        wei_len = ch;
    else
        wei_len = ch*9;
    memcpy(bb, (const void*)(bias), co*sizeof(float));

    if(type == k1s1p0){
        conv_k1s1p0(weight, img, bias, dout, ch, co, ww, wo);
    }
    else if(type == k3s1p1){
        conv_k3s1p1(weight, img, bias, dout, ch, co, ww, wo);
    }
    else if(type == k3s2p0){
        conv_k3s2p0(weight, img, bias, dout, ch, co, ww, wo);
    }
    else if(type == avg) {
        avg_14_1000(img, dout);
    }
}
/*
void convolution(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
    u10 ch, u10 co, u8 ww, u8 wo, u2 k, u2 s, u2 p) {

    u10 i, j, jk, jww, ys, xs;
    s10 xx, yy;
    u8 x, y;
    u2 m, n;
    float acc;
    u16 wei_len = (k==1 ? ch : (ch<<3)+ch);

    float wei[1000+512];
    float bias_t;
    float *bias_buf=wei+wei_len;
    memcpy(bias_buf, (const void*)(bias), co*sizeof(float));

    channel_out:
    for(i=0; i < co; i++){
        memcpy(wei, (const void*)(weight), wei_len*sizeof(float));
        weight += wei_len;
        bias_t = bias[i];
        y_out:
        for(y=0, ys=0; y<wo; y++, ys+=s){
            x_out:
            for(x=0, xs=0; x<wo; x++, xs+=s){
                acc = bias_t;
                channel_in:
                for(j=0, jk=0, jww=0; j < ch; j++, jk+=k, jww+=ww){
#pragma HLS PIPELINE II=1
                    conv_y:
                    for(m=0; m<k; m++){
                        yy = ys + m - p;
                        if(yy>=0 && yy<ww)
                        conv_x:
                        for(n=0; n<k; n++){
                            xx = xs + n - p;
                            if(xx>=0 && xx<ww){
                                acc += wei[(jk + m)*k + n] * img[(jww+yy)*ww + xx];
                            }
                        }
                    }

                }
                *dout++= S_Float(acc) ? 0.0f : acc;
            }
        }
    }
}
*/

