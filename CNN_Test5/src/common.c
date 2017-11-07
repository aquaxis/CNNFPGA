/*
   共通関数
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "common.h"

/*
   drnd()関数
   乱数の生成
*/
double drnd(void)
{
  double rndno; // 生成した乱数

  while((rndno = (double)rand()/RAND_MAX) == 1.0);
  rndno = rndno * 2 - 1;  // -1〜1の間の算数を生成
  return rndno;
}

/*
   f()関数
   伝達関数(シグモイド関数)
 */
double f(double u)
{
  return 1.0 / (1.0 + exp(-u));
}

/*
*/
double getusage(){
  struct rusage usage;
  struct timeval ut;

  getrusage(RUSAGE_SELF, &usage );
  ut = usage.ru_utime;

  return ((double)(ut.tv_sec)*1000 + (double)(ut.tv_usec)*0.001);
}