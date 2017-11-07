#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "cnn.h"
#include "common.h"

/*
   CalcConvolution()関数
   フィルタの適用
 */
double CalcConvolution(
  double *filter,     // フィルタ
  int filter_size,    // フィルタのサイズ
  double *input_data, // 入力データ
  int input_width,    // 入力データの幅
  int input_height,   // 入力データの高さ
  int input_depth,    // 入力データの深さ
  int x, int y        // フィルタの計算位置
)
{
  int m = 0;      // 繰り返し制御用
  int n = 0;      // 繰り返し制御用
  int d = 0;
  double sum = 0; // 総数の値
  int offset;
  int y_start = y - (filter_size / 2);  // フィルタ計算のスタート位置
  int x_start = x - (filter_size / 2);  // フィルタ計算のスタート位置

  for(d = 0; d < input_depth; ++d){
    offset = (input_width * input_height * d);  // 入力データのオフセット
    for(n = 0; n < filter_size; ++n){
      for(m = 0; m < filter_size; ++m){
        sum += input_data[offset + ((y_start + n) * input_width) + (x_start + m)] *
               filter[(n * filter_size) + m];
      }
    }
  }

#if 1
  // 最小値、最大値の計算
  if(sum < 0.0){
    sum = 0.0;
  }
  if(sum > 1.0){
    sum = 1.0;
  }
#endif

  return sum;
}

/*
   Convolution()関数
   畳み込みの計算
 */
#pragma SDS data access_pattern(filter:SEQUENTIAL)
#pragma SDS data access_pattern(input_data:SEQUENTIAL)
#pragma SDS data access_pattern(conv_out:SEQUENTIAL)
#pragma SDS data zero_copy(filter[0:5*5-1])
#pragma SDS data zero_copy(input_data[0:60*60*3-1])
#pragma SDS data zero_copy(conv_out[0:56*56-1])
#pragma SDS data mem_attribute(filter:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(input_data:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(conv_out:PHYSICAL_CONTIGUOUS)
int Convolution
(
  double *filter,     // フィルタ
  int filter_size,    // フィルタのサイズ
  double *input_data, // 入力データ
  int input_width,    // 入力データの幅
  int input_height,   // 入力データの高さ
  int input_depth,    // 入力データの深さ
  double *conv_out,    // 畳み込み結果
  int conv_width,
  int conv_height
)
{
  int x = 0;  // 繰り返し制御用
  int y = 0;  // 繰り返し制御用
  int start_point = filter_size / 2;  // 畳み込み範囲の下限値
//  int conv_width  = input_width  - 2 * (filter_size / 2);  // 畳み込み後の幅
//  int conv_height = input_height - 2 * (filter_size / 2);  // 畳み込み後の高さ

  double buffer0[60*60*3];
  double buffer1[5*5];
  double buffer2[56*56];

// バッファの追加
//#pragma HLS interface ap_memory port=buffer0
//#pragma HLS interface ap_memory port=buffer1
//  memcpy(buffer0, input_data, sizeof(double)*input_width*input_height*input_depth);
//  memcpy(buffer1, filter, sizeof(double)*filter_size*filter_size);

  for(y = 0; y < conv_height; ++y){
    for(x = 0; x < conv_width; ++x){
      conv_out[y * conv_width + x] =
//      buffer2[y * conv_width + x] =
      CalcConvolution(
          filter,
          filter_size,
          input_data,
          input_width,
          input_height,
          input_depth,
          (x + start_point),
          (y + start_point)
        );
    }
  }

// バッファの追加
//#pragma HLS interface ap_memory port=buffer2
//  memcpy(conv_out, buffer2, sizeof(double)*conv_width*conv_height);

  return 0;
}

/*
   MaxPooling()関数
   最大値プーリング
 */
 #pragma SDS data access_pattern(conv_out:SEQUENTIAL)
 #pragma SDS data zero_copy(conv_out[0:56*56-1])
 #pragma SDS data mem_attribute(conv_out:PHYSICAL_CONTIGUOUS)
 double MaxPooling(
  double *conv_out, // 畳み込みデータの出力
  int pool_width,   // プーリング後の幅
  int pool_size,    // プーリングのサイズ
  int x, int y      // 計算位置
)
{
  int m = 0;
  int n = 0;
  double max = -100.0; // 最大値
  double calc = 0;     // 計算値

  for(n = 0; n < pool_size; ++n){
    for(m = 0; m < pool_size; ++m){
      calc = conv_out[(y * pool_width * pool_size * pool_size) +
             (n * pool_width * pool_size) + (x * pool_size) + m];
      if(max < calc) max = calc;
    }
  }

#if 0
  if(max < 0.0){
    max = 0.0;
  }
  if(max > 1.0){
    max = 1.0;
  }
#endif
  return max;
}

/*
   Pooling()関数
   プーリングの計算
 */
#pragma SDS data access_pattern(conv_out:SEQUENTIAL)
#pragma SDS data access_pattern(pool_out:SEQUENTIAL)
#pragma SDS data zero_copy(conv_out[0:56*56-1])
#pragma SDS data zero_copy(pool_out[0:28*28-1])
void Pooling(
  double *conv_out, // 畳み込みデータの入力
  int conv_width,   // 畳み込みデータの幅
  int conv_height,  // 畳み込みデータの高さ
  double *pool_out, // プーリング出力
  int pool_size     // プーリングのサイズ
)
{
  int x = 0;
  int y = 0;
  int pool_width  = conv_width / pool_size;  // プーリング後の幅
  int pool_height = conv_height / pool_size; // プーリング後の高さ

//  double buffer0[56*56];
//  double buffer1[28*28];
//#pragma HLS interface ap_memory port=buffer0
//    memcpy(buffer0, conv_out, sizeof(double)*conv_width*conv_height);

  for(y = 0; y < pool_height; ++y){
    for(x = 0; x < pool_width; ++x){
      pool_out[(y * pool_width) + x] =
        MaxPooling(conv_out, pool_width, pool_size, x, y);
//      buffer1[(y * pool_width) + x] =
//        MaxPooling(conv_out, pool_width, pool_size, x, y);
    }
  }

//#pragma HLS interface ap_memory port=buffer1
//  memcpy(pool_out, buffer1, sizeof(double)*pool_width*pool_height);

}

/*
   CNNLayer()関数
 */
 #pragma SDS data access_pattern(filter:SEQUENTIAL)
 #pragma SDS data access_pattern(input_data:SEQUENTIAL)
 #pragma SDS data access_pattern(conv_out:SEQUENTIAL)
 #pragma SDS data access_pattern(pool_out:SEQUENTIAL)
 #pragma SDS data zero_copy(filter[0:5*5*8-1])
 #pragma SDS data zero_copy(input_data[0:60*60*3-1])
 #pragma SDS data zero_copy(conv_out[0:56*56-1])
 #pragma SDS data zero_copy(pool_out[0:28*28*2-1])
 #pragma SDS data mem_attribute(filter:PHYSICAL_CONTIGUOUS)
 #pragma SDS data mem_attribute(input_data:PHYSICAL_CONTIGUOUS)
 #pragma SDS data mem_attribute(conv_out:PHYSICAL_CONTIGUOUS)
 #pragma SDS data mem_attribute(pool_out:PHYSICAL_CONTIGUOUS)
 int CNNLayer(
  double *filter,     // フィルタ
  int filter_num,     // ふぃるたの数
  int filter_size,    // フィルタのサイズ
  double *input_data, // 入力データ
  int input_width,    // 入力データの幅
  int input_height,   // 入力データの高さ
  int input_depth,    // 入力データの深さ
  double *conv_out,   // 畳み込み結果
  int conv_width,     // 畳み込みデータの幅
  int conv_height,    // 畳み込みデータの高さ
  double *pool_out,   // プーリング出力
  int pool_width,     // プーリング後の幅
  int pool_height,    // プーリング後の高さ
  int pool_size       // プーリングのサイズ
)
{
  int i;
  int offset;

  double buffer00[5*5];
  double buffer01[5*5];
  double buffer02[5*5];
  double buffer03[5*5];
  double buffer04[5*5];
  double buffer05[5*5];
  double buffer06[5*5];
  double buffer07[5*5];
  double buffer10[60*60*3];
  double buffer11[60*60*3];
  double buffer12[28*28*2];
  double buffer13[28*28*2];
  double buffer14[12*12*4];
  double buffer15[12*12*4];
  double buffer16[12*12*4];
  double buffer17[12*12*4];
  double buffer20[56*56];
  double buffer21[56*56];
  double buffer22[24*24];
  double buffer23[24*24];
  double buffer24[8*8];
  double buffer25[8*8];
  double buffer26[8*8];
  double buffer27[8*8];
  double buffer30[28*28];
  double buffer31[28*28];
  double buffer32[12*12];
  double buffer33[12*12];
  double buffer34[4*4];
  double buffer35[4*4];
  double buffer36[4*4];
  double buffer37[4*4];

#pragma HLS interface ap_memory port=buffer00
#pragma HLS interface ap_memory port=buffer01
#pragma HLS interface ap_memory port=buffer02
#pragma HLS interface ap_memory port=buffer03
#pragma HLS interface ap_memory port=buffer04
#pragma HLS interface ap_memory port=buffer05
#pragma HLS interface ap_memory port=buffer06
#pragma HLS interface ap_memory port=buffer07
  if(filter_num >= 0) memcpy(buffer00, &filter[filter_size*filter_size*0], sizeof(double)*filter_size*filter_size);
  if(filter_num >= 1) memcpy(buffer01, &filter[filter_size*filter_size*1], sizeof(double)*filter_size*filter_size);
  if(filter_num >= 2) memcpy(buffer02, &filter[filter_size*filter_size*2], sizeof(double)*filter_size*filter_size);
  if(filter_num >= 3) memcpy(buffer03, &filter[filter_size*filter_size*3], sizeof(double)*filter_size*filter_size);
  if(filter_num >= 4) memcpy(buffer04, &filter[filter_size*filter_size*4], sizeof(double)*filter_size*filter_size);
  if(filter_num >= 5) memcpy(buffer05, &filter[filter_size*filter_size*5], sizeof(double)*filter_size*filter_size);
  if(filter_num >= 6) memcpy(buffer06, &filter[filter_size*filter_size*6], sizeof(double)*filter_size*filter_size);
  if(filter_num >= 7) memcpy(buffer07, &filter[filter_size*filter_size*7], sizeof(double)*filter_size*filter_size);

#pragma HLS interface ap_memory port=buffer10
#pragma HLS interface ap_memory port=buffer11
#pragma HLS interface ap_memory port=buffer12
#pragma HLS interface ap_memory port=buffer13
#pragma HLS interface ap_memory port=buffer14
#pragma HLS interface ap_memory port=buffer15
#pragma HLS interface ap_memory port=buffer16
#pragma HLS interface ap_memory port=buffer17
  if(filter_num >= 0) memcpy(buffer10, input_data, sizeof(double)*input_width*input_height*input_depth);
  if(filter_num >= 1) memcpy(buffer11, input_data, sizeof(double)*input_width*input_height*input_depth);
  if(filter_num >= 2) memcpy(buffer12, input_data, sizeof(double)*input_width*input_height*input_depth);
  if(filter_num >= 3) memcpy(buffer13, input_data, sizeof(double)*input_width*input_height*input_depth);
  if(filter_num >= 4) memcpy(buffer14, input_data, sizeof(double)*input_width*input_height*input_depth);
  if(filter_num >= 5) memcpy(buffer15, input_data, sizeof(double)*input_width*input_height*input_depth);
  if(filter_num >= 6) memcpy(buffer16, input_data, sizeof(double)*input_width*input_height*input_depth);
  if(filter_num >= 7) memcpy(buffer17, input_data, sizeof(double)*input_width*input_height*input_depth);

/*
  for(i = 0; i < filter_num; ++i){
#pragma HLS UNROLL factor=2
#pragma HLS interface ap_memory port=buffer0
#pragma HLS interface ap_memory port=buffer1
    memcpy(buffer0, &filter[filter_size*filter_size*i], sizeof(double)*filter_size*filter_size);
    memcpy(buffer1, input_data, sizeof(double)*input_width*input_height*input_depth);
*/
  // 畳み込みの計算
#pragma HLS PIPELINE
  Convolution(
    buffer00, // filter
    filter_size,
    buffer10, // input_data
    input_width,
    input_height,
    input_depth,
    buffer20, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer20, // conv_out
    conv_width,
    conv_height,
    buffer30, // pool_out
    pool_size
  );

  // 畳み込みの計算
  Convolution(
    buffer01, // filter
    filter_size,
    buffer11, // input_data
    input_width,
    input_height,
    input_depth,
    buffer21, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer21, // conv_out
    conv_width,
    conv_height,
    buffer31, // pool_out
    pool_size
  );

  // 畳み込みの計算
  Convolution(
    buffer02, // filter
    filter_size,
    buffer12, // input_data
    input_width,
    input_height,
    input_depth,
    buffer22, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer22, // conv_out
    conv_width,
    conv_height,
    buffer32, // pool_out
    pool_size
  );

  // 畳み込みの計算
  Convolution(
    buffer03, // filter
    filter_size,
    buffer13, // input_data
    input_width,
    input_height,
    input_depth,
    buffer23, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer23, // conv_out
    conv_width,
    conv_height,
    buffer33, // pool_out
    pool_size
  );

  // 畳み込みの計算
  Convolution(
    buffer04, // filter
    filter_size,
    buffer14, // input_data
    input_width,
    input_height,
    input_depth,
    buffer24, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer24, // conv_out
    conv_width,
    conv_height,
    buffer34, // pool_out
    pool_size
  );

  // 畳み込みの計算
  Convolution(
    buffer05, // filter
    filter_size,
    buffer15, // input_data
    input_width,
    input_height,
    input_depth,
    buffer25, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer25, // conv_out
    conv_width,
    conv_height,
    buffer35, // pool_out
    pool_size
  );

  // 畳み込みの計算
  Convolution(
    buffer06, // filter
    filter_size,
    buffer16, // input_data
    input_width,
    input_height,
    input_depth,
    buffer26, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer26, // conv_out
    conv_width,
    conv_height,
    buffer36, // pool_out
    pool_size
  );

  // 畳み込みの計算
  Convolution(
    buffer07, // filter
    filter_size,
    buffer17, // input_data
    input_width,
    input_height,
    input_depth,
    buffer27, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer27, // conv_out
    conv_width,
    conv_height,
    buffer37, // pool_out
    pool_size
  );

    // プーリングデータ書き込み先のオフセット計算
//    offset = pool_width * pool_height * i;
#pragma HLS interface ap_memory port=buffer30
#pragma HLS interface ap_memory port=buffer31
#pragma HLS interface ap_memory port=buffer32
#pragma HLS interface ap_memory port=buffer33
#pragma HLS interface ap_memory port=buffer34
#pragma HLS interface ap_memory port=buffer35
#pragma HLS interface ap_memory port=buffer36
#pragma HLS interface ap_memory port=buffer37
    if(filter_num >= 0) memcpy(&pool_out[pool_width*pool_height*0], buffer30, sizeof(double)*pool_width*pool_height);
    if(filter_num >= 1) memcpy(&pool_out[pool_width*pool_height*1], buffer31, sizeof(double)*pool_width*pool_height);
    if(filter_num >= 2) memcpy(&pool_out[pool_width*pool_height*2], buffer32, sizeof(double)*pool_width*pool_height);
    if(filter_num >= 3) memcpy(&pool_out[pool_width*pool_height*3], buffer33, sizeof(double)*pool_width*pool_height);
    if(filter_num >= 4) memcpy(&pool_out[pool_width*pool_height*4], buffer34, sizeof(double)*pool_width*pool_height);
    if(filter_num >= 5) memcpy(&pool_out[pool_width*pool_height*5], buffer35, sizeof(double)*pool_width*pool_height);
    if(filter_num >= 6) memcpy(&pool_out[pool_width*pool_height*6], buffer36, sizeof(double)*pool_width*pool_height);
    if(filter_num >= 7) memcpy(&pool_out[pool_width*pool_height*7], buffer37, sizeof(double)*pool_width*pool_height);
//  }


  return 0;
}

/*
   InitFilter()関数
   フィルタを初期化する
 */
#if 0
/*
   乱数フィルタ
*/
void InitFilter(
  double *filter,   // フィルタ
  int filter_size,  // フィルタサイズ
  int filter_num    // フィルタの数
)
{
  int x, y, d;
  int offset;
  for(d = 0; d < filter_num; ++d){
    offset = d * filter_size * filter_size;
    for(y = 0; y < filter_size; ++y){
      for(x = 0; x < filter_size; ++x){
        filter[offset + (y * filter_size) + x] = drnd();
      }
    }
  }
}
#else
/*
   ガボールフィルタ
 */
#define GAMMA (0.7)
#define SIGMA (0.3)
#define PI (3.141592654/180.0)

void InitFilter(
  double *filter,   // フィルタ
  int filter_size,  // フィルタサイズ
  int filter_num    // フィルタの数
)
{
  int x, y, d;

  double nx, ny, xx, yy, w;
  double phai = 0;
  double theta;
  double total;
  double calc;

  int offset;
  for(d = 0; d < filter_num; ++d){
    offset = d * filter_size * filter_size;
    total = 0;
    theta = 360.0 * (((double)d / (double)filter_num) * PI);
    for(y = 0; y < filter_size; ++y){
      for(x = 0; x < filter_size; ++x){
        nx = x * 2 / (double)filter_size - 1;
        ny = y * 2 / (double)filter_size - 1;
        xx =   nx * cos(theta) + ny * sin(theta);
        yy = - nx * sin(theta) + ny * cos(theta);
        w = exp( - (xx * xx + GAMMA * GAMMA * yy * yy ) / (2 * SIGMA * SIGMA));
        calc = w * cos(xx * PI * 2.5 + phai);
        filter[offset + (y * filter_size) + x] = calc;
        total += calc;
      }
    }
    total /= filter_size * filter_size;
    for(y = 0; y < filter_size; ++y){
      for(x = 0; x < filter_size; ++x){
        filter[offset + (y * filter_size) + x] -= total;
      }
    }
  }
}
#endif
