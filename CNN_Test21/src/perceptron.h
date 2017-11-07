#ifndef __PERCEPTRON_HEADER__
#define __PERCEPTRON_HEADER__

// 係数など
#define SEED 65535      /* 乱数のシード*/
#define LIMIT 0.001     /* 誤差の上限値*/
#define BIGNUM 100      /* 誤差の初期値*/
#define HIDDENNO 3      /* 中間層のセル数*/
#define ALPHA  0.3      /* 学習係数*/

double Forward(
  double *input_data,     // 入力層のデータ
  int input_num,          // 入力層の個数
  double *weight_hidden,  // 中間層の重み
  double *weight_out,     // 出力層の重み
  double *hidden_out,     // 中間層の出力
  int hidden_num          // 中間層の数
);

void HiddenLearn(
  double *weight_hidden,  // 中間層の重み
  double *weight_out,     // 出力層の重み
  double *hidden_out,     // 中間層の出力
  int hidden_num,         // 中間層の数
  double *input_data,     // 入力層のデータ
  int input_num,          // 入力層の数
  double teachear,        // 教師情報
  double out              // 出力像の出力
);

void OutLearn(
  double *weight_out, // 出力層の重み
  double *hidden_out, // 中間層の出力
  int hidden_num,     // 中間層の数
  double teacher,     // 教師情報
  double out          // 出力層の出力
);

void InitWeight_Hidden(
  double *weight_hidden,  // 中間層の重み
  int hidden_num,         // 中間層の数
  int input_num           // 入力データの数
);

void InitWeight_Out(
  double *weight_out, // 出力層の重み
  int hidden_num      // 中間層の数
);

#endif
