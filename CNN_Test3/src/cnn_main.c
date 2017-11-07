#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "perceptron.h"
#include "cnn.h"
#include "bitmap.h"
#include "common.h"
#include "cnn_main.h"

// プーリング構造体
typedef struct{
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  double * data;
} PoolImage;

// リストデータ構造体
typedef struct {
  char name[32];
  int teacher;
} InputList;

typedef struct{
  int filter_num;   // フィルター数
  int filter_size;  // フィルターサイズ
  int pool_size;    // プーリングサイズ

  int input_width, input_height, input_depth; // 入力データの幅、高さ、深さ
  int conv_width, conv_height;                // 畳込み後の幅、高さ
  int pool_width, pool_height, pool_depth;    // プーリング後の幅、高さ、深さ

  double * filter;      // フィルタ
  double * input_data;  // 入力データ
  double * conv_out;    // 畳込みデータ
  double * pool_out;    // プーリングデータ
} CNNLayerImage;

/*
   FreeCNNLayerImage()関数
*/
void FreeCNNLayerImage(
  CNNLayerImage * cnn_layer_image
)
{
  int i;

  for(i = 0; i < CNN_LAYER_NUM; ++i){
    sds_free(cnn_layer_image[i].filter);
    sds_free(cnn_layer_image[i].conv_out);
    sds_free(cnn_layer_image[i].pool_out);
  }
  sds_free(cnn_layer_image[0].input_data);
  free(cnn_layer_image);
}

/*
    CNNLayerInit()関数
 */
 CNNLayerImage * CNNLayerInit(
)
{
  CNNLayerImage * cnn_layer_image;

  cnn_layer_image = (CNNLayerImage * )malloc(sizeof(CNNLayerImage) * CNN_LAYER_NUM);

  // CNN Layer 0の設定
  cnn_layer_image[0].filter       = (double *)sds_alloc(sizeof(double) *
                                    CNN_LAYER0_FILTER_SIZE *
                                    CNN_LAYER0_FILTER_SIZE *
                                    CNN_LAYER0_FILTER_NUM);
  cnn_layer_image[0].filter_num   = CNN_LAYER0_FILTER_NUM;
  cnn_layer_image[0].filter_size  = CNN_LAYER0_FILTER_SIZE;
  cnn_layer_image[0].pool_size    = CNN_LAYER0_POOL_SIZE;
  cnn_layer_image[0].input_width  = INPUT_DATA_SIZE;
  cnn_layer_image[0].input_height = INPUT_DATA_SIZE;
  cnn_layer_image[0].input_depth  = INPUT_DATA_DEPTH;
  cnn_layer_image[0].input_data   = (double *)sds_alloc(sizeof(double) *
                                    cnn_layer_image[0].input_width *
                                    cnn_layer_image[0].input_height *
                                    cnn_layer_image[0].input_depth );
  // コンボリューション後の幅
  cnn_layer_image[0].conv_width   = cnn_layer_image[0].input_width  -
                                    (2 * (cnn_layer_image[0].filter_size / 2));
  // コンボリューション後の高さ
  cnn_layer_image[0].conv_height  = cnn_layer_image[0].input_height -
                                    (2 * (cnn_layer_image[0].filter_size / 2));
  // 畳込みの領域確保
  cnn_layer_image[0].conv_out     = (double *)sds_alloc(sizeof(double) *
                                    cnn_layer_image[0].conv_width *
                                    cnn_layer_image[0].conv_height);
  // プーリング後の幅
  cnn_layer_image[0].pool_width   = cnn_layer_image[0].conv_width /
                                    cnn_layer_image[0].pool_size;
  // プーリング後の高さ
  cnn_layer_image[0].pool_height  = cnn_layer_image[0].conv_height /
                                    cnn_layer_image[0].pool_size;
  // プーリング後の深さ
  cnn_layer_image[0].pool_depth   = cnn_layer_image[0].filter_num;
  // プーリングの領域確保
  cnn_layer_image[0].pool_out     = (double *)sds_alloc(sizeof(double) *
                                    cnn_layer_image[0].pool_width *
                                    cnn_layer_image[0].pool_height *
                                    cnn_layer_image[0].pool_depth);

  // CNN Layer 1の設定
  cnn_layer_image[1].filter       = (double *)sds_alloc(sizeof(double) *
                                    CNN_LAYER1_FILTER_SIZE *
                                    CNN_LAYER1_FILTER_SIZE *
                                    CNN_LAYER1_FILTER_NUM);
  cnn_layer_image[1].filter_num   = CNN_LAYER1_FILTER_NUM;
  cnn_layer_image[1].filter_size  = CNN_LAYER1_FILTER_SIZE;
  cnn_layer_image[1].pool_size    = CNN_LAYER1_POOL_SIZE;
  cnn_layer_image[1].input_width  = cnn_layer_image[0].pool_width;
  cnn_layer_image[1].input_height = cnn_layer_image[0].pool_height;
  cnn_layer_image[1].input_depth  = cnn_layer_image[0].pool_depth;
  cnn_layer_image[1].input_data   = cnn_layer_image[0].pool_out;
  // コンボリューション後の幅
  cnn_layer_image[1].conv_width   = cnn_layer_image[1].input_width  -
                                    (2 * (cnn_layer_image[1].filter_size / 2));
  // コンボリューション後の高さ
  cnn_layer_image[1].conv_height  = cnn_layer_image[1].input_height -
                                    (2 * (cnn_layer_image[1].filter_size / 2));
  // 畳込みの領域確保
  cnn_layer_image[1].conv_out     = (double *)sds_alloc(sizeof(double) *
                                    cnn_layer_image[1].conv_width *
                                    cnn_layer_image[1].conv_height);
  // プーリング後の幅
  cnn_layer_image[1].pool_width   = cnn_layer_image[1].conv_width /
                                    cnn_layer_image[1].pool_size;
  // プーリング後の高さ
  cnn_layer_image[1].pool_height  = cnn_layer_image[1].conv_height /
                                    cnn_layer_image[1].pool_size;
  // プーリング後の深さ
  cnn_layer_image[1].pool_depth   = cnn_layer_image[1].filter_num;
  // プーリングの領域確保
  cnn_layer_image[1].pool_out     = (double *)sds_alloc(sizeof(double) *
                                    cnn_layer_image[1].pool_width *
                                    cnn_layer_image[1].pool_height *
                                    cnn_layer_image[1].pool_depth);

  // CNN Layer 2の設定
  cnn_layer_image[2].filter       = (double *)sds_alloc(sizeof(double) *
                                    CNN_LAYER2_FILTER_SIZE *
                                    CNN_LAYER2_FILTER_SIZE *
                                    CNN_LAYER2_FILTER_NUM);
  cnn_layer_image[2].filter_num   = CNN_LAYER2_FILTER_NUM;
  cnn_layer_image[2].filter_size  = CNN_LAYER2_FILTER_SIZE;
  cnn_layer_image[2].pool_size    = CNN_LAYER2_POOL_SIZE;
  cnn_layer_image[2].input_width  = cnn_layer_image[1].pool_width;
  cnn_layer_image[2].input_height = cnn_layer_image[1].pool_height;
  cnn_layer_image[2].input_depth  = cnn_layer_image[1].pool_depth;
  cnn_layer_image[2].input_data   = cnn_layer_image[1].pool_out;
  // コンボリューション後の幅
  cnn_layer_image[2].conv_width   = cnn_layer_image[2].input_width  -
                                    (2 * (cnn_layer_image[2].filter_size / 2));
  // コンボリューション後の高さ
  cnn_layer_image[2].conv_height  = cnn_layer_image[2].input_height -
                                    (2 * (cnn_layer_image[2].filter_size / 2));
  // 畳込みの領域確保
  cnn_layer_image[2].conv_out     = (double *)sds_alloc(sizeof(double) *
                                    cnn_layer_image[2].conv_width *
                                    cnn_layer_image[2].conv_height);
  // プーリング後の幅
  cnn_layer_image[2].pool_width   = cnn_layer_image[2].conv_width /
                                    cnn_layer_image[2].pool_size;
  // プーリング後の高さ
  cnn_layer_image[2].pool_height  = cnn_layer_image[2].conv_height /
                                    cnn_layer_image[2].pool_size;
  // プーリング後の深さ
  cnn_layer_image[2].pool_depth   = cnn_layer_image[2].filter_num;
  // プーリングの領域確保
  cnn_layer_image[2].pool_out     = (double *)sds_alloc(sizeof(double) *
                                    cnn_layer_image[2].pool_width *
                                    cnn_layer_image[2].pool_height *
                                    cnn_layer_image[2].pool_depth);

  return cnn_layer_image;
}

/*
   CNNLayer()関数
   CNNを行う
*/
int execCNN(
  CNNLayerImage *cnn_layer_image
)
{
  int i;

  // CNN Layer 処理
  for(i = 0; i < CNN_LAYER_NUM; ++i){
    CNNLayer(
      cnn_layer_image[i].filter,
      cnn_layer_image[i].filter_num,
      cnn_layer_image[i].filter_size,
      cnn_layer_image[i].input_data,
      cnn_layer_image[i].input_width,
      cnn_layer_image[i].input_height,
      cnn_layer_image[i].input_depth,
      cnn_layer_image[i].conv_out,
      cnn_layer_image[i].conv_width,
      cnn_layer_image[i].conv_height,
      cnn_layer_image[i].pool_out,
      cnn_layer_image[i].pool_width,
      cnn_layer_image[i].pool_height,
      cnn_layer_image[i].pool_size
    );
  }

  return 0;
}

/*
   CNN()関数
   教師データからCNN学習を行って全結合層（パーセプロトロン）の
   中間層、出力層の重みを算出する
 */
int CNN(
  InputList *input_image,
  int num_of_input_data,
  double *weight_hidden,
  double *weight_out,
  int learnmode
)
{
  double *hidden_data;  // 中間層データ
  double out;           // 最終出力
  int count = 0;        // 学習回数
  double err = BIGNUM;  // 誤差
  int i;

  int input_width, input_height, input_depth; // 入力データの幅、高さ、深さ

  int pool_out_num;
  Image *image;
  int d,x,y;
  int src_pt, dst_pt;

  // 時間測定用
  double st, et;
  double usage = 0.0;
  int usage_count = 0;

  // 学習時に全結合の出力層の重み退避用
  double * weight_out_old = (double *)malloc(sizeof(double) * (HIDDEN_NUM + 1));

  CNNLayerImage * cnn_layer_image;
  cnn_layer_image = CNNLayerInit(); // CNN Layer 情報の初期化
  // 全結合の中間層の出力データの領域確保
  hidden_data = (double *)malloc(sizeof(double) * HIDDEN_NUM);

  // フィルタの初期化
  InitFilter(cnn_layer_image[0].filter, CNN_LAYER0_FILTER_SIZE, CNN_LAYER0_FILTER_NUM);
  InitFilter(cnn_layer_image[1].filter, CNN_LAYER1_FILTER_SIZE, CNN_LAYER1_FILTER_NUM);
  InitFilter(cnn_layer_image[2].filter, CNN_LAYER2_FILTER_SIZE, CNN_LAYER2_FILTER_NUM);

  // メインループ
  while(1){
    err = 0.0;
    for(i = 0; i < num_of_input_data; i++){  // 学習データ毎の繰り返し
      // 画像データ読込み
      if(!learnmode) printf("File: %s(%d)\n", input_image[i].name, input_image[i].teacher);
      image = ReadBMP(input_image[i].name);
      input_width  = image->width;
      input_height = image->height;
      input_depth  = image->bpp/8;
      // uunsigned char → double変換
      for(d = 0; d < input_depth; ++d){
        for(y = 0; y < input_height; ++y){
          for(x = 0; x < input_width; ++x){
            dst_pt = (input_height * input_width * d) + (input_width * y) + x;
            src_pt = (input_width * input_depth * y) + (input_depth * x) + d;
            cnn_layer_image[0].input_data[dst_pt] =
              (double)image->data[src_pt] / 255.0;
          }
        }
      }

      // 畳み込み＋プーリング
      if(!learnmode) st = getusage(); // CNN開始時刻の取得

      execCNN(cnn_layer_image);

      // 全結合の入力個数の計算
      pool_out_num = cnn_layer_image[CNN_LAYER_NUM-1].pool_width *
                     cnn_layer_image[CNN_LAYER_NUM-1].pool_height *
                     cnn_layer_image[CNN_LAYER_NUM-1].pool_depth;
      // 全結合(パーセプトロン)
      out = Forward(
        cnn_layer_image[CNN_LAYER_NUM-1].pool_out, pool_out_num,
        weight_hidden, weight_out, hidden_data, HIDDEN_NUM);

      if(!learnmode) et = getusage(); // CNN終了時刻の取得
      if(!learnmode){
        usage += et -st;
        ++usage_count;
      }

        // 全結合の重みの学習
      if(learnmode){
        // 出力層の重みの退避
        memcpy(weight_out_old, weight_out, sizeof(double) * HIDDEN_NUM);
        // 出力層の重みの調整
        OutLearn(weight_out, hidden_data, HIDDEN_NUM, (double)input_image[i].teacher, out);
        // 中間層の重みの計算
        HiddenLearn(weight_hidden, weight_out_old, hidden_data, HIDDEN_NUM,
                    cnn_layer_image[CNN_LAYER_NUM-1].pool_out, pool_out_num,
                    (double)input_image[i].teacher, out);
        // 誤差計算
        err += (out - (double)input_image[i].teacher) *
               (out - (double)input_image[i].teacher);
      }

      FreeImg(image);

      if(!learnmode) printf("[Answer] %lf\n", out);
    }
    if(learnmode) printf("[Learn Out] %d: %f\n", count, err);
    ++count;
    if(((err < LIMIT) && learnmode) || !learnmode) break;
  }
  if(!learnmode){
    printf("UsageTIme: %6.3lf[ms]\n", usage / usage_count);
  }

  free(hidden_data);
  free(weight_out_old);
  FreeCNNLayerImage(cnn_layer_image);

  return 0;
}

/*
   メイン関数
 */
int main(int argc, char **argv)
{
  double *weight_hidden;          // 全結合の中間層の重み
  double *weight_out;             // 全結合の出力層の重み
  int num_of_input_data = 0;      // 入力画像の数
  int learnmode = 0;              // 処理モード(0:演算モード, 1:学習モード)
  char *filename;                 // ファイル名
  FILE *fp;                       // ファイルポインタ
  InputList input_list[MAX_LIST]; // リストデータ

  srand(SEED);  // 乱数の初期化

  printf("CNN - Start\n");

  // モード判定
  if(argc > 1){
    if(!strcmp(argv[1], "-l")){
      learnmode = 1;
    }
  }

  // ファイルの選択
  if(learnmode){
    // 学習モード時
    filename = list_learn;
    printf("Mode: Learn\n");
  }else{
    // テスト時
    if(argc > 1){
      filename = argv[1];
      printf("Mode: Custom\n");
    }else{
      filename = list_test;
      printf("Mode: Test\n");
    }
  }
  printf("List File: %s\n", filename);

  // 重みの領域確保
  weight_hidden = (double *)malloc(sizeof(double) * HIDDEN_NUM * (POOL_OUT_NUM + 1));
  weight_out    = (double *)malloc(sizeof(double) * (HIDDEN_NUM + 1));

  if(learnmode){
    // 学習モード時
    // 重みの初期化
    InitWeight_Hidden(weight_hidden, HIDDEN_NUM, POOL_OUT_NUM);
    InitWeight_Out(weight_out, HIDDEN_NUM);
  }else{
    // 演算モード時
    // 中間層の重みの保存
    if((fp=fopen("weight_hidden.bin", "r+")) != NULL){
      fread(weight_hidden, sizeof(double) * HIDDEN_NUM * (POOL_OUT_NUM + 1), 1, fp);
    }
    fclose(fp);
    // 出力層の重みの保存
    if((fp=fopen("weight_out.bin", "r+")) != NULL){
      fread(weight_out, sizeof(double) * (HIDDEN_NUM + 1), 1, fp);
    }
    fclose(fp);
  }

  // 画像リストの読込み
  if((fp=fopen(filename, "r")) != NULL){
    // ファイルが終わるまで読み込む
    while( fscanf(fp,"%s %d",
                  &input_list[num_of_input_data].name[0],
                  &input_list[num_of_input_data].teacher) != EOF
    ){
      ++num_of_input_data;
    }
  }
  fclose(fp);

  printf("Num of Input Data: %d\n", num_of_input_data);

  // CNN
  CNN(input_list, num_of_input_data, weight_hidden, weight_out, learnmode);

  // 学習した重みの保存
  if(learnmode){
    // 中間層の重みの保存
    if((fp=fopen("weight_hidden.bin", "w+")) != NULL){
      fwrite(weight_hidden, sizeof(double) * HIDDEN_NUM * (POOL_OUT_NUM + 1), 1, fp);
    }
    fclose(fp);
    // 出力層の重みの保存
    if((fp=fopen("weight_out.bin", "w+")) != NULL){
      fwrite(weight_out, sizeof(double) * (HIDDEN_NUM + 1), 1, fp);
    }
    fclose(fp);

    // テスト画像リストの読込み
    num_of_input_data = 0;
    if((fp=fopen(list_test, "r")) != NULL){
      // ファイルが終わるまで読み込む
      while( fscanf(fp,"%s %d",
                    &input_list[num_of_input_data].name[0],
                    &input_list[num_of_input_data].teacher) != EOF
      ){
        ++num_of_input_data;
      }
    }
    fclose(fp);

    // 学習テスト
    CNN(input_list, num_of_input_data, weight_hidden, weight_out, 0);
  }

  // メモリ解放
  free(weight_hidden);
  free(weight_out);

  return 0;

}
