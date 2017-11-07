#ifndef __CNN_HEADER__
#define __CNN_HEADER__

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
);

void Pooling(
  double *conv_out, // 畳み込みデータの入力
  int conv_width,   // 畳み込みデータの幅
  int conv_height,  // 畳み込みデータの高さ
  double *pool_out, // プーリング出力
  int pool_size     // プーリングのサイズ
);

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
);

double execCNN2(
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
  int pool_size2,

  int pool_depth2,

  double *weight_hidden,
  double *weight_out,
  double *hidden_data
);

double execCNN3(
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
  int pool_size2,

  double *weight_hidden,
  double *weight_out,
  double *hidden_data
);

#endif
