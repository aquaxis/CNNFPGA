#ifndef __CNN_MAIN_HEADER__
#define __CNN_MAIN_HEADER__

#define MAX_LIST (256)

#define INPUT_DATA_SIZE (60)
#define INPUT_DATA_DEPTH (3)

#define CNN_LAYER_NUM   (3)

#define CNN_LAYER0_FILTER_NUM  (2)
#define CNN_LAYER0_FILTER_SIZE (5)
#define CNN_LAYER0_POOL_SIZE   (2)

#define CNN_LAYER1_FILTER_NUM  (4)
#define CNN_LAYER1_FILTER_SIZE (5)
#define CNN_LAYER1_POOL_SIZE   (2)

#define CNN_LAYER2_FILTER_NUM  (8)
#define CNN_LAYER2_FILTER_SIZE (5)
#define CNN_LAYER2_POOL_SIZE   (2)

#define POOL_OUT_NUM (4*4*CNN_LAYER2_FILTER_NUM)
#define HIDDEN_NUM (POOL_OUT_NUM/2)

char *list_learn = "list_learn.txt";
char *list_test  = "list_test.txt";

#endif
