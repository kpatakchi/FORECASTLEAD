#!/bin/sh

#SBATCH --job-name=datecheck2
#SBATCH --output=LOGS/datecheck2.out
#SBATCH --error=LOGS/datecheck2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --mail-user=k.patakchi.yousefi@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --account=esmtst

export TZ=UTC  # Force UTC timezone to avoid DST transition issues

start="20211030 13"

echo "Step 4) Create datasets for each YYYYMMDD and temporal resolution (hourly, three-hourly, and six-hourly)"
((($(date -d "${subdate}" +%s) - $(date -d "${current_date}" +%s))/86400 + 1))
i1="+ 90 hours"
i2="+ 1 hour"
i3="+ 24 hours"

i11="+ 144 hours"
i22="+ 3 hours"

i111="+ 240 hours"
i222="+ 6 hours"

source /p/project/cesmtst/patakchiyousefi1/CODES-MS3/FORECASTLEAD/bashenv

start=$(date -d "$start" "+%Y%m%d %H")
end=$(date -d "$start + 1 month" "+%Y%m%d %H")

echo "Start date: $start"
echo "End date<=: $end"

current_date=$start
echo "Current date: $current_date"
echo "Start of while loop from ${current_date} to < ${end}"

while (( $(date -d "${current_date}" +%s) < $(date -d "${end}" +%s) )); do

    ####################### 0-90-1 ######################

    search_patterns=()  # Move outside the inner loop to accumulate patterns for the day
    subdate=$current_date
    echo "Start of while loop from $(date -d "${subdate}") to < $(date -d "${current_date} ${i1}")"
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i1}" +%s) )); do
        sub_formatted_date=$(date -d "${subdate}" "+%Y%m%d%H")
        daycount=$(printf "%02d" $((($(date -d "${subdate}" +%s) - $(date -d "${current_date}" +%s) - 3600)/86400 + 1)))
        newfile=$(ls $HRES_DUMP3/*day*$daycount*${sub_formatted_date}*)
        if [ -n "$newfile" ]; then
            search_patterns+=("$newfile")
        else
            echo "--------------- No file found for: $sub_formatted_date daycount $daycount -------------------"
        fi
        subdate=$(date -d "${subdate} ${i2}")
        echo "Updated subdate: $subdate"
    done
    echo "Number of files to process with CDO: ${#search_patterns[@]}"

    # Print all contents of search_patterns
    echo "Files in search_patterns:"
    for file in "${search_patterns[@]}"; do
        echo "$file"
    done

    subdate=$(date -d "${subdate} - 1 hour")
    echo "Subdate after subtracting 1 hour: $subdate"
    subdate=$(date -d "${subdate} ${i22}")
    echo "Subdate after applying i22: $subdate"

    echo "0-90 is done"
    sleep 5
    ####################### 90-144-3 ######################

    search_patterns=()  # Move outside the inner loop to accumulate patterns for the day
    echo "Start of while loop from $(date -d "${subdate}") to < $(date -d "${current_date} ${i11}")"
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i11}" +%s) )); do
        sub_formatted_date=$(date -d "${subdate}" "+%Y%m%d%H")
        daycount=$(printf "%02d" $((($(date -d "${subdate}" +%s) - $(date -d "${current_date}" +%s) - 10800)/86400 + 1)))
        newfile=$(ls $HRES_DUMP3/*day*$daycount*${sub_formatted_date}*)
        if [ -n "$newfile" ]; then
            search_patterns+=("$newfile")
        else
            echo "--------------- No file found for: $sub_formatted_date daycount $daycount -------------------"
        fi
        subdate=$(date -d "${subdate} ${i22}")
        echo "Updated subdate: $subdate"
    done
    echo "Number of files to process with CDO: ${#search_patterns[@]}"
    
    # Print all contents of search_patterns
    echo "Files in search_patterns:"
    for file in "${search_patterns[@]}"; do
        echo "$file"
    done
    
    subdate=$(date -d "${subdate} - 3 hours")
    echo "Subdate after subtracting 1 hour: $subdate"

    subdate=$(date -d "${subdate} ${i222}")
    echo "Subdate after applying i22: $subdate"

    echo "90-144 is done"
    sleep 5

    ####################### 144-240-6 ######################
    
    search_patterns=()  # Move outside the inner loop to accumulate patterns for the day
    echo "Start of while loop from $(date -d "${subdate}") to < $(date -d "${current_date} ${i111}")"
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i111}" +%s) )); do
        sub_formatted_date=$(date -d "${subdate}" "+%Y%m%d%H")
        daycount=$(printf "%02d" $((($(date -d "${subdate}" +%s) - $(date -d "${current_date}" +%s) - 21600)/86400 + 1)))
        newfile=$(ls $HRES_DUMP3/*day*$daycount*${sub_formatted_date}*)
        if [ -n "$newfile" ]; then
            search_patterns+=("$newfile")
        else
            echo "--------------- No file found for: $sub_formatted_date daycount $daycount -------------------"
        fi
        subdate=$(date -d "${subdate} ${i222}")
    done
    echo "Number of files to process with CDO: ${#search_patterns[@]}"
    # Print all contents of search_patterns
    echo "Files in search_patterns:"
    for file in "${search_patterns[@]}"; do
        echo "$file"
    done

    echo $current_date "processed!"
    current_date=$(date -d "${current_date} ${i3}")
    echo "current_date after subtracting i3: $current_date"
    sleep 5

done