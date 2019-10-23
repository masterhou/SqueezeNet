//
//  hls_cnn.cpp
//  SqueezeNet
//
//  Created by Thomas Hou on 2019/10/14.
//  Copyright © 2019 Thomas Hou. All rights reserved.
//

#include "hls_cnn.h"

typedef struct df
{
    float a;
    float b;
}df;


float wei[64*3*3];
float bb[1000];
float im[113*64*3]; // stride + 2*padding = 3
static u10 wei_len, wx3;
static u16 w2;

void conv_k3s1p1(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
u10 ch, u10 co, u6 ww, u6 wo) {
#pragma HLS ARRAY_PARTITION variable=wei cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=im cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=acc_t cyclic factor=2
#pragma HLS INLINE
    u2 m, n;
    u8 x, y, j;
    u10 i;
    u12 yj;
    s8 xx, yy;
    u16 yw;
    u18 wl, dl;
    float buf[9];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1

    wei_len = ch*9;

    volatile float *imp = img-ww; // for padding=1
    k3s1p1_y:
    for(y=0; y<wo; y++, imp+=ww, dout+=ww){
        for(j=0, yj=0, yw=0; j < ch; j++, yj+=wx3, yw+=w2){
            memcpy(im+yj, (void*)(imp+yw), wx3*sizeof(float)); // k=3;s=1
        }
        conv_k3s1p1:
        for(i=0, wl=0, dl=0; i < co; i++, wl+=wei_len, dl+=w2){
            memcpy(wei, (const void*)(weight+wl), wei_len*sizeof(float));
            float bbi = bb[i];
            volatile float *out= dout+dl;
            k3s1p1_x:
            for(x=0; x<wo; x++){
                u16 wi = 0;
                float acc = bbi;
                float *pm = im;
                k3s1p1_ch:
                for(j=0, yj=0; j < ch; j++, yj+=wx3){
#pragma HLS UNROLL skip_exit_check factor=2
#pragma HLS PIPELINE II=1
                    u4 _i=0;
                    k3s1p1_kernel_y1:
                    for(m=0, yy=y-1; m<3; m++, yy++, pm+=ww){
                        k3s1p1_kernel_x1:
                        for(n=0, xx=x-1; n<3; n++, xx++, wi++){
                            if(xx>=0 && xx<ww && yy>=0 && yy<ww){
                                buf[_i++] = wei[wi] * pm[xx];
                            }
                            else {
                                buf[_i++] = 0.f;
                            }
                        }
                    }
                    acc += buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7] + buf[8];
                }
                out[x] = S_Float(acc) ? 0.0f : acc;
            }
        }
    }
}


void conv_k3s2p0(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
u10 ch, u10 co, u8 ww, u8 wo){
#pragma HLS ARRAY_PARTITION variable=wei cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=im cyclic factor=2
#pragma HLS INLINE
    u2 m, n, j; // max = 3
    u8 x, y, i;
    u12 yj;
    u19 yw;

    k3s2p0_yo:
    for(y=0; y<113; y++, img+=227*2, dout+=113){
        k3s2p0_img_cpy:
        for(j=0, yj=0, yw=0; j < 3; j++, yj+=227*3, yw+=227*227){
            memcpy(im+yj, (void*)(img+yw), 227*3*sizeof(float)); // k=3;s=2
        }
        k3s2p0_co:
        for(i=0; i < 64; i++){
            float bbi = bb[i];
            volatile float *out= dout+113*113*i;
            memcpy(wei, (const void*)(weight+27*i), 27*sizeof(float));
            k3s2p0_xo:
            for(x=0; x<113; x++){
#pragma HLS PIPELINE II=2
                u8 wi = 0;
                float *pm = im + (x<<1);
                float acc = bbi;
                k3s2p0_ch:
                for(j=0; j < 3; j++){
                    k3s2p0_y2:
                    for(m=0; m<3; m++, pm+=227){
                        k3s2p0_x2:
                        for(n=0; n<3; n++){
                            acc += wei[wi++] * pm[n];
                        }
                    }
                }
                *out++ = S_Float(acc) ? 0.0f : acc;
            }
        }
    }
}

void conv_k1s1p0(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
u10 ch, u10 co, u6 ww, u6 wo) {
//#pragma HLS ARRAY_PARTITION variable=wei cyclic factor=2
//#pragma HLS ARRAY_PARTITION variable=im cyclic factor=2
#pragma HLS INLINE
    u8 x, y;
    u10 i, j;
    u13 yj;
    u19 yw,wl;

    k1s1p0_y_out:
    for(y=0; y<wo; y++, img+=ww, dout+=ww){
        for(j=0, yj=0, yw=0; j < ch; j++, yj+=ww, yw+=w2){
            memcpy(im+yj, (void*)(img+yw), ww*sizeof(float));
        }
        k1s1p0_co:
        for(i=0, yw=0, wl=0; i < co; i++, yw+=w2, wl+=ch){
            memcpy(wei, (const void*)(weight+wl), ch*sizeof(float));
            // weight += ch;
            float bbi = bb[i];
            volatile float *out= dout+yw;
            k1s1p0_x_out:
            for(x=0; x<wo; x++){
                float acc = bbi;
                k1s1p0_k:
                for(j=0, yj=x; j < ch; j++, yj+=ww){
#pragma HLS PIPELINE II=4
#pragma HLS UNROLL skip_exit_check factor=16
                    acc += wei[j] * im[yj];
                }
                *out++ = S_Float(acc) ? 0.0f : acc;
            }
        }
    }
}

void max_k3s2(volatile float *weight, volatile float *img, volatile float *bias, volatile float *dout,
u10 ch, u10 co, u8 ww, u6 wo) {
#pragma HLS ARRAY_PARTITION variable=im cyclic factor=2
#pragma HLS INLINE
    u2 m, n; // max = 3
    u8 x, y; // out // wo
    u8 xx, yy;
    u10 i, mm;
    u20 yw, dw;

    max_y:
    for(y=0; y<wo; y++, img+=ww*2, dout+=wo){
#pragma HLS LOOP_TRIPCOUNT min=56 max=56 avg=56
        max_img_cpy_co:
        for(i=0, yw=0, dw=0; i < co; i++, yw+=w2, dw+=wo*wo){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
            memcpy(im, (void*)(img+yw), wx3*sizeof(float)); // k=3;s=2

            volatile float *out = dout + dw;
            max_x:
            for (x=0; x<wo; x++) {
#pragma HLS LOOP_TRIPCOUNT min=56 max=56 avg=56
#pragma HLS PIPELINE II=2
                float val = 0.0f;
                for (m=0, yy=(y<<1), mm=0; m<3; m++, yy++, mm+=ww) {
                    for (n=0, xx=(x<<1); n<3; n++, xx++) {
                        if(xx<ww && yy<ww) {
                            float t = im[ mm + xx];
                            if(t>val)
                                val = t;
                        }
                    }
                }
                *out++ = val;
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
        float v = 0.0f;
        for(u8 k=0; k<14*14; k++)
            v += *img++;
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
    else if(type == max_k3){
        max_k3s2(weight, img, bias, dout, ch, co, ww, wo);
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


typedef union f_i {
    float f;
    uint32_t i;
}f_i;

void u64_mul(u64 *weight, u64 *img, u64 *bias, u64 *dout,
        u10 ch, u10 co, u8 ww, u8 wo) {
#pragma HLS INTERFACE m_axi depth=1024 port=weight offset=slave
#pragma HLS INTERFACE m_axi depth=1024 port=img offset=slave
#pragma HLS INTERFACE m_axi depth=1024 port=bias offset=slave
#pragma HLS INTERFACE m_axi depth=1024 port=dout offset=slave

    for(u10 i=0; i<100; i++) {
#pragma HLS PIPELINE II=1
        f_i a, b;
        a.i = (bias[i]<<32)>>32;// (u32)(bias[i]&0xFFFFFFFFL); //bias[i].range(31, 0);
        b.i = bias[i]>>32; //bias[i].range(63,32);
        bb[i*2  ]=a.f;
        bb[i*2+1]=b.f;
    }

    for(u10 i=0; i<50; i++) {
#pragma HLS PIPELINE II=1
        f_i a, b;
        a.f = bb[i*4+0]* bb[i*4+1];
        b.f = bb[i*4+2]* bb[i*4+3];
        //u64 res = a;
        //res.range(31,  0) = a;
        //res.range(63, 32) = b;
        dout[i] = (((u64)b.i)<<32 | a.i); //res;
    }


    //conv_k3s1p1(weight, img, bias, dout, ch, co, ww, wo);
}
