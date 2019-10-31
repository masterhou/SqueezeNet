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


void copy(float *buf0, float *buf1, float *in, u10 size) {
#pragma HLS INLINE
    double_copy:
    for(u10 a=0; a<(size); a++) {
#pragma HLS LOOP_TRIPCOUNT min=140 max=140
#pragma HLS PIPELINE II=1
        *buf0++ = in[a];
        *buf1++ = in[a];
    }
}

float wei[64*3*3];
float bb[1000];
float im[227*3*3];
static u10 wx3;
static u16 w2;

void conv_k3s1p1(float *weight, float *img, float *bias, float *dout,
u7 ch, u9 co, u6 ww, u6 wo) {
#pragma HLS ARRAY_PARTITION variable=wei cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=im cyclic factor=2
#pragma HLS INLINE
    u2 m, n;
    u6 x, y;
    u7 j;
    u9 i;
    u12 yj;
    s7 xx, yy;
    u16 yw;
    u18 wl, dl;
    float im2[56*16*3], im3[56*16*3];
    
    float buf0[9];
#pragma HLS ARRAY_PARTITION variable=buf0 complete dim=1
    float buf1[9];
#pragma HLS ARRAY_PARTITION variable=buf1 complete dim=1

    u10 wei_len = ch*9;

    float *imp = img-ww; // for padding=1
    k3s1p1_y:
    for(y=0; y<ww; y++, imp+=ww, dout+=wo) {
#pragma HLS LOOP_TRIPCOUNT min=56 max=56 avg=56
        for(j=0, yj=0, yw=0; j < ch; j++, yj+=wx3, yw+=w2){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16 avg=16
            //memcpy(im+yj, (void*)(imp+yw), wx3*sizeof(float)); // k=3;s=1
            copy(im2+yj, im3+yj, (imp+yw), wx3);
        }
        conv_k3s1p1:
        for(i=0, wl=0, dl=0; i < co; i++, wl+=wei_len, dl+=wo*wo){
#pragma HLS LOOP_TRIPCOUNT min=64 max=64 avg=64
            memcpy(wei, (const void*)(weight+wl), wei_len*sizeof(float));
            float bbi = bb[i];
            float *out= dout+dl;
            k3s1p1_x:
            for(x=0; x<ww; x+=2){
#pragma HLS LOOP_TRIPCOUNT min=56/2 max=56/2
                u10 wi = 0;
                float acc0 = bbi, acc1 = bbi;
                k3s1p1_ch:
                for(j=0, yj=0; j < ch; j++){
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS UNROLL skip_exit_check factor=2
#pragma HLS PIPELINE II=1
                    u4 _i=0;
                    k3s1p1_kernel_y1:
                    for(m=0, yy=y-1; m<3; m++, yy++, yj+=ww){
                        k3s1p1_kernel_x1:
                        for(n=0, xx=x-1; n<3; n++, xx++, wi++, _i++){
                            if(xx<0 || xx==ww || yy<0 || yy==ww)
                                buf0[_i] = 0.f;
                            else
                                buf0[_i] = wei[wi] * im2[yj+xx];

                            if(xx+1==ww || yy<0 || yy==ww)
                                buf1[_i] = 0.f;
                            else
                                buf1[_i] = wei[wi] * im3[yj+xx+1];
                        }
                    }
                    acc0 = acc0 + buf0[0] + buf0[1] + buf0[2] + buf0[3] + buf0[4] + buf0[5] + buf0[6] + buf0[7] + buf0[8];
                    acc1 = acc1 + buf1[0] + buf1[1] + buf1[2] + buf1[3] + buf1[4] + buf1[5] + buf1[6] + buf1[7] + buf1[8];
                }
                *out++ = S_Float(acc0) ? 0.0f : acc0;
                *out++ = S_Float(acc1) ? 0.0f : acc1;
            }
        }
    }
}


void conv_k3s2p0(float *weight, float *img, float *bias, float *dout,
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
            float *out= dout+113*113*i;
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
void conv_k1s1p0(float *weight, float *img, float *bias, float *dout,
u10 ch, u10 co, u6 ww, u6 wo) {
//#pragma HLS ARRAY_PARTITION variable=wei cyclic factor=2
//#pragma HLS ARRAY_PARTITION variable=im cyclic factor=2
#pragma HLS INLINE
    u6 x, y;
    u10 i, j;
    u13 yj;
    u19 yw,wl;
    float im0[56*128], im1[56*128];

    k1s1p0_y_out:
    for(y=0; y<ww; y++, img+=ww, dout+=wo){
        for(j=0, yj=0, yw=0; j < ch; j++, yj+=ww, yw+=w2){
            //memcpy(im+yj, (void*)(img+yw), ww*sizeof(float));
            copy(im0+yj, im1+yj, (img+yw), ww);
        }
        k1s1p0_co:
        for(i=0, yw=0, wl=0; i < co; i++, yw+=wo*wo, wl+=ch){
            memcpy(wei, (const void*)(weight+wl), ch*sizeof(float));
            float bbi = bb[i];
            float *out= dout+yw;
            k1s1p0_x_out:
            for(x=0; x<ww; x+=2){
                float acc0 = bbi, acc1 = bbi;
                k1s1p0_k:
                for(j=0, yj=x; j < ch; j+=1, yj+=ww){
#pragma HLS PIPELINE II=4
#pragma HLS UNROLL skip_exit_check factor=16
                    acc0 += wei[j] * im0[yj];
                    acc1 += wei[j] * im1[yj+1];
                }
                *out++ = S_Float(acc0) ? 0.0f : acc0;
                *out++ = S_Float(acc1) ? 0.0f : acc1;
            }
        }
    }
}

void max_k3s2(float *weight, float *img, float *bias, float *dout,
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
            float *out = dout + dw;
            max_x:
            for (x=0; x<wo; x++) {
#pragma HLS LOOP_TRIPCOUNT min=56 max=56 avg=56
#pragma HLS PIPELINE II=2
                float val = 0.0f;
                for (m=0, yy=(y<<1), mm=0; m<3; m++, yy++, mm+=ww) {
                    for (n=0, xx=(x<<1); n<3; n++, xx++) {
                        float t = im[ mm + xx];
                        if(t>val)
                            val = t;
                    }
                }
                *out++ = val;
            }
        }
    }
}

void avg_14_1000(float *img, float *dout){
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

void add(float *weight, float *img, float *bias, float *dout,
    u10 ch, u10 co, u8 ww, u8 wo, u2 type) {
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
void convolution(float *weight, float *img, float *bias, float *dout,
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
