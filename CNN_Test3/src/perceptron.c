#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "perceptron.h"
#include "common.h"

/*
   Forward()関数
   順方向の計算
 */
double Forward(
  double *input_data,     // 入力層のデータ
  int input_num,          // 入力層の個数
  double *weight_hidden,  // 中間層の重み
  double *weight_out,     // 出力層の重み
  double *hidden_out,     // 中間層の出力
  int hidden_num          // 中間層の数
)
{
  int i = 0;
  int j = 0;
  double sum; // 歪み付き総数の値
  double out; // 出力値

  // 中間層の計算
  for(i = 0; i < hidden_num; ++i){
    sum = 0.0;  // 総数の初期化
    for(j = 0; j < input_num; ++j){
      sum += input_data[j] * weight_hidden[i * (input_num + 1) + j];
    }
    sum -= weight_hidden[i * (input_num + 1) + j];  // 閾値の処理
    hidden_out[i] = f(sum);
  }

  // 出力層の計算
  out = 0.0;
  for(i = 0; i < hidden_num; ++i){
    out += hidden_out[i] * weight_out[i];
  }
  out -= weight_out[i]; // 閾値の処理

  return f(out);
}

/*
   HiddenLearn()関数
   中間層の重み学習
 */
void HiddenLearn(
  double *weight_hidden,  // 中間層の重み
  double *weight_out,     // 出力層の重み
  double *hidden_out,     // 中間層の出力
  int hidden_num,         // 中間層の数
  double *input_data,     // 入力層のデータ
  int input_num,          // 入力層の数
  double teachear,        // 教師データ
  double out              // 出力像の出力
)
{
  int i = 0;
  int j = 0;
  double dj; // 中間層の重み計算に利用

  for(j = 0; j < hidden_num; ++j){  // 中間層の各セルjを対象
    dj = hidden_out[j] * (1 - hidden_out[j]) * weight_out[j] *
         (teachear - out) * out * (1 - out);
    for(i = 0; i < input_num; ++i){  // i番目の重みを処理
      weight_hidden[(j * (input_num + 1)) + i] += ALPHA * input_data[i] * dj;
    }
    weight_hidden[(j * (input_num + 1)) + i] += ALPHA * (-1.0) * dj;  // 閾値の学習
  }
}

/*
   OutLearn()関数
   出力層の重み学習
 */
void OutLearn(
  double *weight_out, // 出力層の重み
  double *hidden_out, // 中間層の出力
  int hidden_num,     // 中間層の数
  double teacher,     // 教師データ
  double out          // 出力層の出力
)
{
  int i = 0;
  double d; // 重み計算に利用

  d = (teacher - out) * out * (1 - out); // 誤差の計算
  for(i = 0; i < hidden_num; ++i){
    weight_out[i] += ALPHA * hidden_out[i] * d; // 重みの学習
  }
  weight_out[i] += ALPHA * (-1.0) * d;  // 閾値の学習
}

/*
   Initweight_hidden()関数
   中間層の重みの初期化
 */
void InitWeight_Hidden(
  double *weight_hidden,  // 中間層の重み
  int hidden_num,         // 中間層の数
  int input_num           // 入力データの数
)
{
  int i = 0;
  int j = 0;

  // 乱数による重みの決定
  for(i = 0; i < hidden_num; ++i){
    for(j = 0; j < input_num + 1; ++j){
      weight_hidden[(i * (input_num + 1)) + j] = drnd();
    }
  }
}

/*
   Initweight_out()関数
   出力層の重みの初期化
 */
void InitWeight_Out(
  double *weight_out, // 出力層の重み
  int hidden_num      // 中間層の数
)
{
  int i = 0;

  // 乱数による重みの決定
  for(i = 0; i< hidden_num + 1; ++i){
    weight_out[i] = drnd();
  }
}
