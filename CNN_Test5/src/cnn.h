#ifndef __CNN_HEADER__
#define __CNN_HEADER__

int Convolution
(
  double *filter,     // フィルタ
  int filter_size,    // フィルタのサイズ
  double *input_data, // 入力データ
  int input_width,    // 入力データの幅
  int input_height,   // 入力データの高さ
  int input_depth,    // 入力データの深さ
  double *conv_out    // 畳み込み結果
);

void Pooling(
  double *conv_out, // 畳み込みデータの入力
  int conv_width,   // 畳み込みデータの幅
  int conv_height,  // 畳み込みデータの高さ
  double *pool_out, // プーリング出力
  int pool_size     // プーリングのサイズ
);

void InitFilter(
  double *filter,  // フィルタ
  int filter_size,        // フィルタサイズ
  int input_depth         // 入力データの深さ
);

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
);

#endif
