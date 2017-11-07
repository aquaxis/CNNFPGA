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
#pragma HLS UNROLL
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
//#pragma SDS data access_pattern(filter:SEQUENTIAL)
//#pragma SDS data access_pattern(input_data:SEQUENTIAL)
//#pragma SDS data access_pattern(conv_out:SEQUENTIAL)
//#pragma SDS data zero_copy(filter[0:5*5-1])
//#pragma SDS data zero_copy(input_data[0:60*60*3-1])
//#pragma SDS data zero_copy(conv_out[0:56*56-1])
//#pragma SDS data mem_attribute(filter:PHYSICAL_CONTIGUOUS)
//#pragma SDS data mem_attribute(input_data:PHYSICAL_CONTIGUOUS)
//#pragma SDS data mem_attribute(conv_out:PHYSICAL_CONTIGUOUS)
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
#pragma HLS UNROLL
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
// #pragma SDS data access_pattern(conv_out:SEQUENTIAL)
// #pragma SDS data zero_copy(conv_out[0:56*56-1])
// #pragma SDS data mem_attribute(conv_out:PHYSICAL_CONTIGUOUS)
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
//#pragma SDS data access_pattern(conv_out:SEQUENTIAL)
//#pragma SDS data access_pattern(pool_out:SEQUENTIAL)
//#pragma SDS data zero_copy(conv_out[0:56*56-1])
//#pragma SDS data zero_copy(pool_out[0:28*28-1])
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
#pragma HLS UNROLL
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
// #pragma SDS data access_pattern(filter:SEQUENTIAL)
// #pragma SDS data access_pattern(input_data:SEQUENTIAL)
// #pragma SDS data access_pattern(conv_out:SEQUENTIAL)
// #pragma SDS data access_pattern(pool_out:SEQUENTIAL)
// #pragma SDS data zero_copy(filter[0:5*5*8-1])
// #pragma SDS data zero_copy(input_data[0:60*60*3-1])
// #pragma SDS data zero_copy(conv_out[0:56*56-1])
// #pragma SDS data zero_copy(pool_out[0:28*28*2-1])
// #pragma SDS data mem_attribute(filter:PHYSICAL_CONTIGUOUS)
// #pragma SDS data mem_attribute(input_data:PHYSICAL_CONTIGUOUS)
// #pragma SDS data mem_attribute(conv_out:PHYSICAL_CONTIGUOUS)
// #pragma SDS data mem_attribute(pool_out:PHYSICAL_CONTIGUOUS)
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

  double buffer0[5*5];
  double buffer1[60*60*3];
  double buffer2[56*56];
  double buffer3[28*28];

  for(i = 0; i < filter_num; ++i){
//#pragma HLS UNROLL factor=2
//#pragma HLS interface ap_memory port=buffer0
//#pragma HLS interface ap_memory port=buffer1
//    memcpy(buffer0, &filter[filter_size*filter_size*i], sizeof(double)*filter_size*filter_size);
//    memcpy(buffer1, input_data, sizeof(double)*input_width*input_height*input_depth);
#pragma HLS PIPELINE
    // 畳み込みの計算
    Convolution(
    	&filter[filter_size*filter_size*i],
//      buffer0,
      filter_size,
      input_data,
//      buffer1,
      input_width,
      input_height,
      input_depth,
//      conv_out,
      buffer2,
      conv_width,
      conv_height
    );
    // プーリングデータ書き込み先のオフセット計算
    offset = pool_width * pool_height * i;
    // プーリングの計算
    Pooling(
//      conv_out,
      buffer2,
      conv_width,
      conv_height,
      &pool_out[offset],
//      buffer3,
      pool_size
    );
//#pragma HLS interface ap_memory port=buffer3
//    memcpy(&pool_out[offset], buffer3, sizeof(double)*pool_width*pool_height);
  }


  return 0;
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
 int CNNLayer0(
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

  double buffer_l00[5*5];
  double buffer_l01[5*5];
  double buffer_l10[60*60*3];
  double buffer_l11[60*60*3];
  double buffer_l20[56*56];
  double buffer_l21[56*56];
  double buffer_l30[28*28];
  double buffer_l31[28*28];

#pragma HLS interface ap_memory port=buffer_l00
#pragma HLS interface ap_memory port=buffer_l01
  memcpy(buffer_l00, &filter[filter_size*filter_size*0], sizeof(double)*filter_size*filter_size);
  memcpy(buffer_l01, &filter[filter_size*filter_size*1], sizeof(double)*filter_size*filter_size);

#pragma HLS interface ap_memory port=buffer_l10
#pragma HLS interface ap_memory port=buffer_l11
  memcpy(buffer_l10, input_data, sizeof(double)*input_width*input_height*input_depth);
  memcpy(buffer_l11, input_data, sizeof(double)*input_width*input_height*input_depth);

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
    buffer_l00, // filter
    filter_size,
    buffer_l10, // input_data
    input_width,
    input_height,
    input_depth,
    buffer_l20, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer_l20, // conv_out
    conv_width,
    conv_height,
//    &pool_out[pool_width*pool_height*0],
    buffer_l30, // pool_out
    pool_size
  );

  // 畳み込みの計算
  Convolution(
    buffer_l01, // filter
    filter_size,
    buffer_l11, // input_data
    input_width,
    input_height,
    input_depth,
    buffer_l21, // conv_out
    conv_width,
    conv_height
  );
  // プーリングの計算
  Pooling(
    buffer_l21, // conv_out
    conv_width,
    conv_height,
//    &pool_out[pool_width*pool_height*1],
    buffer_l31, // pool_out
    pool_size
  );


    // プーリングデータ書き込み先のオフセット計算
//    offset = pool_width * pool_height * i;
#pragma HLS interface ap_memory port=buffer_l30
#pragma HLS interface ap_memory port=buffer_l31
    memcpy(&pool_out[pool_width*pool_height*0], buffer_l30, sizeof(double)*pool_width*pool_height);
    memcpy(&pool_out[pool_width*pool_height*1], buffer_l31, sizeof(double)*pool_width*pool_height);
//  }


  return 0;
}

/*
   CNNLayer()関数
   CNNを行う
*/
#pragma SDS data access_pattern(filter0:SEQUENTIAL)
#pragma SDS data access_pattern(input_data0:SEQUENTIAL)
#pragma SDS data access_pattern(conv_out0:SEQUENTIAL)
#pragma SDS data access_pattern(pool_out0:SEQUENTIAL)
#pragma SDS data zero_copy(filter0[0:5*5*2-1])
#pragma SDS data zero_copy(input_data0[0:60*60*3-1])
#pragma SDS data zero_copy(conv_out0[0:56*56-1])
#pragma SDS data zero_copy(pool_out0[0:28*28*2-1])

#pragma SDS data access_pattern(filter1:SEQUENTIAL)
#pragma SDS data access_pattern(input_data1:SEQUENTIAL)
#pragma SDS data access_pattern(conv_out1:SEQUENTIAL)
#pragma SDS data access_pattern(pool_out1:SEQUENTIAL)
#pragma SDS data zero_copy(filter1[0:5*5*4-1])
#pragma SDS data zero_copy(input_data1[0:56*56*2-1])
#pragma SDS data zero_copy(conv_out1[0:14*14-1])
#pragma SDS data zero_copy(pool_out1[0:12*12*4-1])

#pragma SDS data access_pattern(filter2:SEQUENTIAL)
#pragma SDS data access_pattern(input_data2:SEQUENTIAL)
#pragma SDS data access_pattern(conv_out2:SEQUENTIAL)
#pragma SDS data access_pattern(pool_out2:SEQUENTIAL)
#pragma SDS data zero_copy(filter2[0:5*5*8-1])
#pragma SDS data zero_copy(input_data2[0:12*12*4-1])
#pragma SDS data zero_copy(conv_out2[0:8*8-1])
#pragma SDS data zero_copy(pool_out2[0:4*4*8-1])

#pragma SDS data access_pattern(weight_hidden:SEQUENTIAL)
#pragma SDS data access_pattern(weight_out:SEQUENTIAL)
#pragma SDS data access_pattern(hidden_data:SEQUENTIAL)
#pragma SDS data zero_copy(weight_hidden[0:4*4*8-1])
#pragma SDS data zero_copy(weight_out[0:64-1])
#pragma SDS data zero_copy(hidden_data[0:64-1])
double execCNN(
  double *filter0,
  int filter_num0,     int filter_size0,    double *input_data0,
  int input_width0,    int input_height0,   int input_depth0,
  double *conv_out0,   int conv_width0,     int conv_height0,
  double *pool_out0,   int pool_width0,     int pool_height0,
  int pool_size0,

  double *filter1,
  int filter_num1,     int filter_size1,    double *input_data1,
  int input_width1,    int input_height1,   int input_depth1,
  double *conv_out1,   int conv_width1,     int conv_height1,
  double *pool_out1,   int pool_width1,     int pool_height1,
  int pool_size1,

  double *filter2,
  int filter_num2,     int filter_size2,    double *input_data2,
  int input_width2,    int input_height2,   int input_depth2,
  double *conv_out2,   int conv_width2,     int conv_height2,
  double *pool_out2,   int pool_width2,     int pool_height2,
  int pool_size2
)
{
  double out;
  int pool_out_num = 0;

  double buffer00[5*5*2];
  double buffer01[5*5*4];
  double buffer02[5*5*8];
  double buffer1[60*60*3];
  double buffer20[56*56];
  double buffer21[24*24];
  double buffer22[8*8];
  double buffer30[28*28*2];
  double buffer31[12*12*4];
  double buffer32[4*4*8];

  #pragma HLS interface ap_memory port=buffer00
  #pragma HLS interface ap_memory port=buffer01
  #pragma HLS interface ap_memory port=buffer02
  #pragma HLS interface ap_memory port=buffer1
  #pragma HLS interface ap_memory port=buffer31
  //  memcpy(buffer00, filter0, sizeof(double)*filter_size0*filter_size0*filter_num0);
  memcpy(buffer01, filter1, sizeof(double)*filter_size1*filter_size1*filter_num1);
  memcpy(buffer02, filter2, sizeof(double)*filter_size2*filter_size2*filter_num2);
//  memcpy(buffer1, input_data0, sizeof(double)*input_width0*input_height0*input_depth0);

  CNNLayer0(
    filter0, // filter0,
//    buffer00, // filter0,
    filter_num0,
    filter_size0,
    input_data0, // input_data0,
//    buffer1, // input_data0,
    input_width0,
    input_height0,
    input_depth0,
    buffer20, // conv_out0,
    conv_width0,
    conv_height0,
    pool_out0, // pool_out0,
//    buffer30, // pool_out0,
    pool_width0,
    pool_height0,
    pool_size0
  );

  CNNLayer(
    buffer01, // filter1,
    filter_num1,
    filter_size1,
    pool_out0, // input_data1,
//    buffer30, // input_data1,
    input_width1,
    input_height1,
    input_depth1,
    buffer21, // conv_out1,
    conv_width1,
    conv_height1,
    buffer31, // pool_out1,
    pool_width1,
    pool_height1,
    pool_size1
  );

  CNNLayer(
    buffer02, // filter2,
    filter_num2,
    filter_size2,
    buffer31, // input_data2,
    input_width2,
    input_height2,
    input_depth2,
    buffer22, // conv_out2,
    conv_width2,
    conv_height2,
    buffer32, // pool_out2,
    pool_width2,
    pool_height2,
    pool_size2
  );

#pragma HLS interface ap_memory port=buffer32
    memcpy(pool_out2, buffer32, sizeof(double)*pool_width2*pool_height2*filter_num2);

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
