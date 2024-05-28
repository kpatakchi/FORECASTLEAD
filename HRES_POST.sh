#!/bin/sh

source bashenv

#mkdir $PSCRATCH_DIR $PSCRATCH_DIR2
#mkdir $HRES_PREP $HRES_DUMP $HRES_DUMP2 $HRES_DUMP3 $HRES_DUMP4
#rm -r $HRES_DUMP/* $HRES_DUMP2/* $HRES_DUMP3/* $HRES_DUMP4/* $HRES_LOG/CDO* 
#mkdir $HRES_POST

echo "Step 1) Split merged leadtime files into individual years and months using CDO splityearmon"
 for leadtimefile in $(ls $PREDICT_FILES)
 do
    cdo -O -b F64 --no_history splityearmon $PREDICT_FILES/$leadtimefile $HRES_DUMP/$leadtimefile &>> $HRES_LOG/CDO_post_1.out
done

echo "Step 2) Split files into individual days using CDO splitday"
 for file in $(ls $HRES_DUMP)
 do
    date_part=$(echo $file | grep -oP 'day\d+.*\K\d{6}' | grep -oP '\d{6}')
    day_part=$(echo $file | grep -oP 'day\d+')    
    cdo -O -b F64 --no_history splitday $HRES_DUMP/$file $HRES_DUMP2/ADAPTER_DE05.${day_part}.merged.${date_part} &>> $HRES_LOG/CDO_post_2.out
done

echo "Step 3) Split files into individual hours using CDO splithour"
 for file in $(ls $HRES_DUMP2)
 do
    date_part=$(echo $file | grep -oP 'day\d+.*\K\d{8}' | grep -oP '\d{8}')
    day_part=$(echo $file | grep -oP 'day\d+')    
    cdo -O -b F64 --no_history splithour $HRES_DUMP2/$file $HRES_DUMP3/ADAPTER_DE05.${day_part}.merged.${date_part} &>> $HRES_LOG/CDO_post_3.out
    echo $date_part $day_part
done
