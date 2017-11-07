/*
   BITMAP画像 読み書き
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "bitmap.h"

/*
   ReadBMP()関数
   BITMAPファイルを読み込む
 */
Image * ReadBMP(char *filename)
{
  uint32_t width, height, bpp;
  Image *img;
  FILE *fp;
  uint8_t header[HEADERSIZE];
  uint32_t offset_of_data;
  uint32_t imgsize;

  // ファイルオープン
  if((fp = fopen(filename, "rb")) == NULL){
    fprintf(stderr, "Error: %s could not open.\n", filename);
    return NULL;
  }

  // ヘッダーの読込み
  fread(header, sizeof(uint8_t), HEADERSIZE, fp);

  // BITMAP判定
  if(strncmp((char *)header, "BM", 2)){
    fprintf(stderr, "Error: %s int not Bitmap file.\n", filename);
    fclose(fp);
    return NULL;
  }

  memcpy(&offset_of_data,  header + 10, sizeof(offset_of_data));
  memcpy(&width,  header + 18, sizeof(width));
  memcpy(&height, header + 22, sizeof(height));
  memcpy(&bpp,  header + 28, sizeof(bpp));

  // bit/pixelの検査
  if(bpp != 24){
    fprintf(stderr, "Error: %s is not 24bit bpp.\n", filename);
  }

  imgsize = width * height * (bpp / 8); // イメージのサイズ

  // イメージ用構造体の領域確保
  if((img = (Image *)malloc(sizeof(Image))) == NULL){
    fprintf(stderr, "Error: Allocation error\n");
    return NULL;
  }

  // イメージの領域確保
  if((img->data = (uint8_t *)malloc(imgsize)) == NULL){
    fprintf(stderr, "Error: Allocation error\n");
    free(img);
    return NULL;
  }

  img->width  = width;
  img->height = width;
  img->bpp    = bpp;

  // イメージの読込み
  fseek(fp, offset_of_data, SEEK_SET);
  fread(img->data, sizeof(uint8_t), imgsize, fp);

  fclose(fp);

  return img;
}

/*
   WriteBMP()関数
   BITMAPを書き込む
 */
int WriteBMP(char *filename, Image *img)
{
  FILE *fp;
  uint8_t header[HEADERSIZE];
  uint32_t file_size;
  uint32_t offset_of_data;
  uint32_t info_header_size;
  uint32_t planes;
  uint32_t img_size;
  int32_t xppm, yppm;

  // ファイルオープン
	if((fp = fopen(filename, "wb+")) == NULL){
		fprintf(stderr, "Error: %s could not open.", filename);
		return -1;
	}
  img_size          = img->width * img->height * ( img->bpp / 8 );
  file_size         = img_size + HEADERSIZE;
  offset_of_data    = HEADERSIZE;
  info_header_size  = INFOHEADERSIZE;
  planes            = 1;
  xppm              = 1;
  yppm              = 1;

  memset(header, 0, HEADERSIZE);

  // ヘッダ生成
  header[0] = 'B';
  header[1] = 'M';
  memcpy(header +  2, &file_size, sizeof(file_size));
  memcpy(header + 10, &offset_of_data, sizeof(offset_of_data));
  memcpy(header + 14, &info_header_size, sizeof(info_header_size));
  memcpy(header + 18, &img->width, sizeof(img->width));
  memcpy(header + 22, &img->height, sizeof(img->height));
  memcpy(header + 26, &planes, sizeof(planes));
  memcpy(header + 28, &img->bpp, sizeof(img->bpp));
  memcpy(header + 34, &img_size, sizeof(img_size));
  memcpy(header + 38, &xppm, sizeof(xppm));
  memcpy(header + 42, &yppm, sizeof(yppm));

  fwrite(&header, sizeof(uint8_t), HEADERSIZE, fp); // ヘッダの書込み
  fwrite(img->data, sizeof(uint8_t), img_size, fp); // イメージの書込み

  fclose(fp);

  return 0;
}

/*
   FreeImg()関数
   イメージ用構造体を開放する
 */
void FreeImg(Image *img)
{
  free(img->data);
  free(img);
}

#if 0
// for Debug
void main(int argc, unsigned char **argv)
{
  Image *img;
  uint32_t width, height, bpp;

  img = ReadBMP(argv[1]);
  WriteBMP("dst.bmp", img);

  FreeImg(img);
}
#endif
