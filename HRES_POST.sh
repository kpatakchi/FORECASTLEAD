#!/bin/sh

echo "Step 1) Split merged leadtime files into individual years and months using CDO splityearmon"
 for leadtimefile in $(ls $PREDICT_FILES)
 do
    cdo -O -b F64 --no_history splityearmon $PREDICT_FILES/$leadtimefile $HRES_DUMP/$leadtimefile
done

echo "Step 2) Split files into individual days using CDO splitday"
 for file in $(ls $HRES_DUMP)
 do
    date_part=$(echo $file | grep -oP 'day\d+.*\K\d{6}' | grep -oP '\d{6}')
    day_part=$(echo $file | grep -oP 'day\d+')    
    echo "Processing date: $date_part, day: $day_part"
    cdo -O -b F64 --no_history splitday $HRES_DUMP/$file $HRES_DUMP2/ADAPTER_DE05.${day_part}.merged.corrected.${date_part}
done

echo "Step 3) Split files into individual hours using CDO splithour"
 for file in $(ls $HRES_DUMP2)
 do
    date_part=$(echo $file | grep -oP 'day\d+.*\K\d{8}' | grep -oP '\d{8}')
    day_part=$(echo $file | grep -oP 'day\d+')    
    echo "Processing date: $date_part, day: $day_part"
    cdo -O -b F64 --no_history splithour $HRES_DUMP2/$file $HRES_DUMP3/ADAPTER_DE05.${day_part}.merged.corrected.${date_part}
    echo $date_part $day_part
done

rm -r $HRES_DUMP $HRES_DUMP2
