#ifndef __BITMAP_HEADER__
#define __BITMAP_HEADER__

#define FILEHEADERSIZE (14)
#define INFOHEADERSIZE (40)
#define HEADERSIZE (FILEHEADERSIZE+INFOHEADERSIZE)

// イメージ用構造体
typedef struct{
  uint32_t width;   // イメージの幅
  uint32_t height;  // イメージの高さ
  uint32_t bpp;     // イメージのbit/pixel
  uint8_t * data;   // イメージデータ
} Image;

Image * ReadBMP(char *filename);
int WriteBMP(char *filename, Image *img);
void FreeImg(Image *img);

#endif
