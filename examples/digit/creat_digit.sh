DATA_ROOT=../../data/digit/
BACKEND=lmdb
EXAMPLE=./
LM_DIR_TRAIN=./digit_lmdb_train
LM_DIR_TEST=./digit_lmdb_test

rm -rf $LM_DIR_TEST
rm -rf $LM_DIR_TRAIN

if [ -d ${DATA_ROOT} ];then
  echo "transmitting training data..."
  ../../build/tools/convert_imageset \
      --backend=$BACKEND --gray --shuffle \
      $DATA_ROOT \
      $DATA_ROOT/train.txt \
      $LM_DIR_TRAIN
else
  echo "DATA_ROOT DIR not exists..."
fi


if [ -d ${DATA_ROOT} ];then
  echo "transmitting testing data..."
  ../../build/tools/convert_imageset \
    --backend=$BACKEND --check_size --gray --shuffle \
      $DATA_ROOT \
      $DATA_ROOT/test.txt \
      $LM_DIR_TEST
else
  echo "DATA_ROOT DIR not exists..."
fi
