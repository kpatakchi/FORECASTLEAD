#!/bin/sh
# each day takes 2mins to run (60minutes for a month)

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <start_time>"
    exit 1
fi
start="$1"

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv

echo "Step 4) Create datasets for each YYYYMMDD and temporal resolution (hourly, three-hourly, and six-hourly)"

i1="+ 90 hours"
i2="+ 1 hour"
i3="+ 24 hours"

i11="+ 144 hours"
i22="+ 3 hours"

i111="+ 240 hours"
i222="+ 6 hours"


start=$(date -d "$start" "+%Y%m%d %H")
end=$(date -d "$start + 1 month" "+%Y%m%d %H")

current_date=$start
while (( $(date -d "${current_date}" +%s) <= $(date -d "${end}" +%s) )); do

    ####################### 0-90-1 ######################

    search_patterns=()  # Move outside the inner loop to accumulate patterns for the day
    subdate=$current_date
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i1}" +%s) )); do
        sub_formatted_date=$(date -d "${subdate}" "+%Y%m%d%H")
        daycount=$(printf "%02d" $((($(date -d "${subdate}" +%s) - $(date -d "${current_date}" +%s))/86400 + 1)))
        newfile=$(ls $HRES_DUMP3/*day*$daycount*${sub_formatted_date}*)
        if [ -n "$newfile" ]; then
            search_patterns+=("$newfile")
        fi
        subdate=$(date -d "${subdate} ${i2}")
    done
    echo "Number of files to process with CDO: ${#search_patterns[@]}"

    #first one starts from 13, second one is a copy with zero values over 12, third is 1 and 2 merged
    merged_output="$HRES_DUMP4/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").13.0-90-1.boundary_1.tp.nc"
    merged_output2="$HRES_DUMP4/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").12.0-90-zero.nc"
    merged_output3="$HRES_POST/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").12.0-90-1.boundary_1.tp.nc"
    
    cdo -O -b F64 --no_history -timcumsum -divc,1000. -setattribute,tp@units=m -chname,pr,tp -mergetime ${search_patterns[@]} $merged_output
    cdo -O -b F64 --no_history -mulc,0 -seltimestep,1 -shifttime,-1hour -copy $merged_output $merged_output2
    cdo -O -b F64 --no_history -mergetime $merged_output2 $merged_output $merged_output3 
    ncatted -O -h -a history,global,o,c,"post-processed to ECMWF format (reverted from instantaneous to cumulative total precipitation) - by k.patakchi.yousefi" $merged_output3 
    
    rm $merged_output $merged_output2
    subdate=$(date -d "${subdate} - 1 hour")
    
    echo "0-90 is done"
    ####################### 90-144-3 ######################

    search_patterns=()  # Move outside the inner loop to accumulate patterns for the day
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i11}" +%s) )); do
        sub_formatted_date=$(date -d "${subdate}" "+%Y%m%d%H")
        daycount=$(printf "%02d" $((($(date -d "${subdate}" +%s) - $(date -d "${current_date}" +%s))/86400 + 1)))
        newfile=$(ls $HRES_DUMP3/*day*$daycount*${sub_formatted_date}*)
        if [ -n "$newfile" ]; then
            search_patterns+=("$newfile")
        fi
        subdate=$(date -d "${subdate} ${i22}")
    done
    echo "Number of files to process with CDO: ${#search_patterns[@]}"

    #first one starts from 13, second one is a copy with zero values over 12, third is 1 and 2 merged
    merged_output11="$HRES_DUMP4/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").13.90-144-3.boundary_1.tp.nc"
    merged_output22="$HRES_DUMP4/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").12.90-144-prev.nc"
    merged_output33="$HRES_POST/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").12.90-144-3.boundary_1.tp.nc"

    echo "cdo 1:"
    cdo -L -O -b F64 --no_history -seltimestep,2/19 -divc,1000. -setattribute,tp@units=m -chname,pr,tp -mergetime ${search_patterns[@]} $merged_output11

    echo "cdo 2:"
    cdo -O -b F64 --no_history -seltimestep,91 -copy $merged_output3 $merged_output22 
    
    echo "cdo 3:"
    cdo -O -b F64 --no_history -timcumsum -mergetime $merged_output22 $merged_output11 $merged_output33 

    echo "nco:"
    ncatted -O -h -a history,global,o,c,"post-processed to ECMWF format (reverted from instantaneous to cumulative total precipitation) - by k.patakchi.yousefi" $merged_output33 

    rm $merged_output11 $merged_output22
    subdate=$(date -d "${subdate} - 3 hours")

    echo "90-144 is done"

    ####################### 144-240-6 ######################
    
    search_patterns=()  # Move outside the inner loop to accumulate patterns for the day
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i111}" +%s) )); do
        sub_formatted_date=$(date -d "${subdate}" "+%Y%m%d%H")
        daycount=$(printf "%02d" $((($(date -d "${subdate}" +%s) - $(date -d "${current_date}" +%s))/86400 + 1)))
        newfile=$(ls $HRES_DUMP3/*day*$daycount*${sub_formatted_date}*)
        if [ -n "$newfile" ]; then
            search_patterns+=("$newfile")
        fi
        subdate=$(date -d "${subdate} ${i222}")
    done
    echo "Number of files to process with CDO: ${#search_patterns[@]}"
    
    #first one starts from 13, second one is a copy with zero values over 12, third is 1 and 2 merged
    merged_output111="$HRES_DUMP4/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").13.144-240-6.boundary_1.tp.nc"
    merged_output222="$HRES_DUMP4/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").12.144-240-prev.nc"
    merged_output333="$HRES_POST/ADAPTER_DE05_$(date -d "${current_date}" "+%Y%m%d").12.144-240-6.boundary_1.tp.nc"
    
    cdo -L -O -b F64 --no_history -seltimestep,2/17 -divc,1000. -setattribute,tp@units=m -chname,pr,tp -mergetime ${search_patterns[@]} $merged_output111 
    cdo -O -b F64 --no_history -seltimestep,19 -copy $merged_output33 $merged_output222 
    cdo -O -b F64 --no_history -timcumsum -mergetime $merged_output222 $merged_output111 $merged_output333 
    ncatted -O -h -a history,global,o,c,"post-processed to ECMWF format (reverted from instantaneous to cumulative total precipitation) - by k.patakchi.yousefi" $merged_output333 

    rm $merged_output111 $merged_output222

    echo $current_date "processed!"
    current_date=$(date -d "${current_date} ${i3}")
done