#!/bin/bash

SCENE=("k18_00" "k18_01" "k18_02" "k18_03" "k18_04" "k19_00" "k19_01" "k19_06" "k19_07" "k19_09" "k19_11" "k19_12" "k19_13" "k19_14")

len=${#SCENE[@]}
for((i=0; i<$len; i++ ))
do


  DATA_DIR=/home/lh/all_datasets/kitti_scene/"${SCENE[i]}"/
  bash /home/lh/zipnerf/scripts/local_colmap_kitti.sh ${DATA_DIR}
done
