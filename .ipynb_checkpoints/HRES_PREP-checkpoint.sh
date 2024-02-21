#!/bin/sh

#SBATCH --job-name=HRES_PREP
#SBATCH --output=LOGS/HRES_PREP.out
#SBATCH --error=LOGS/HRES_PREP.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=03:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=deepacf

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv

#mkdir $PSCRATCH_DIR $PSCRATCH_DIR2
#mkdir $HRES_PREP $HRES_DUMP $HRES_DUMP2 $HRES_DUMP3 $HRES_DUMP4
rm -r $HRES_PREP/* $HRES_DUMP/* $HRES_DUMP2/* $HRES_DUMP3/* $HRES_DUMP4/* $HRES_LOG/CDO* 

echo "0. Copying/extracting original hres files from original directory to scratch"
#rm -r $HRES_OR/*
#cp $HRES_RET/2018/*.nc $HRES_OR/
#cp $HRES_RET/2019/*.nc $HRES_OR/
#cp $HRES_RET/2020/*.nc $HRES_OR/
#cp $HRES_RET/2021/*.nc $HRES_OR/
#cp $HRES_RET/2022/*.nc $HRES_OR/
#cp $HRES_RET/2023/*.nc $HRES_OR/

echo "done!"

echo "Step 1) deltat, mulc, unit, selname"
for ncfile in $(ls $HRES_OR)
do
    cdo -O -b F64 --no_history -deltat -mulc,1000. -setattribute,pr@units=mm -chname,tp,pr -selname,tp $HRES_OR/$ncfile $HRES_DUMP/$ncfile.deltat.mulc.unit.chname.selname.nc &>> $HRES_LOG/CDO.out
done
echo "done!"

echo "Step 2) Merge all three forecast files by date"
# Assuming the dates are in YYYYMMDD format
for date in $(ls $HRES_DUMP | grep -oP '\d{8}' | sort -u)
do
    # Merging files for each date
    cdo -O -b F64 --no_history mergetime $(ls $HRES_DUMP/ADAPTER_DE05_$date.*.boundary_1.nc.deltat.mulc.unit.chname.selname.nc) \
                     $HRES_DUMP2/ADAPTER_DE05_$date.merged.nc &>> $HRES_LOG/CDO_2.out
done
echo "done!"


echo "Step 3) Pick the forecast data per day for each date"
for ncfile in $(ls $HRES_DUMP2)
do
    
    cdo -O -b F64 --no_history -seltimestep,1/24 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day01.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,25/48 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day02.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,49/72 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day03.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,73/92 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day04.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,93/100 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day05.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,101/108 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day06.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,109/112 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day07.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,113/116 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day08.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,117/120 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day09.nc &>> $HRES_LOG/CDO_3.out
    cdo -O -b F64 --no_history -seltimestep,121/124 $HRES_DUMP2/$ncfile $HRES_DUMP3/$ncfile.day10.nc &>> $HRES_LOG/CDO_3.out

    
done
echo "done!"

echo "Step 4) Merge day and ref files (day1, day2, day3, ..., day10)"
# Loop through each day number
for day in {01..10}
do
    # Merging day files for each day number, (IMPORTANT: sorted by date)
    cdo -O -b F64 --no_history mergetime $(ls $HRES_DUMP3/ADAPTER_DE05_*.day$day.nc | sort -t_ -k4,4) \
                     $HRES_DUMP4/ADAPTER_DE05.day$day.merged.nc &>> $HRES_LOG/CDO_5.out    
done
echo "done!"

echo "Step 5) Add history attribute"
for day in {01..10}
do
    # Add changes to history attribute
    ncatted -O -h -a history,global,o,c,"deltat, mulc, unit, selname, group files by forecast lead time - by k.patakchi.yousefi" $HRES_DUMP4/ADAPTER_DE05.day$day.merged.nc $HRES_PREP/ADAPTER_DE05.day$day.merged.nc
done
echo "done!"

#rm -r $HRES_DUMP/* $HRES_DUMP2/* $HRES_DUMP3/*
