#!/bin/sh
# Simplified script for debugging date progression

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <start_time>"
    exit 1
fi
start="$1"

echo "Starting date processing simulation..."

i1="+ 90 hours"
i2="+ 1 hour"
i3="+ 24 hours"

i11="+ 144 hours"
i22="+ 3 hours"

i111="+ 240 hours"
i222="+ 6 hours"

start=$(date -d "$start" "+%Y%m%d %H")
end=$(date -d "$start + 1 month" "+%Y%m%d %H")

echo "start date: $start"
echo "end date <=: $end"

current_date=$start
while (( $(date -d "${current_date}" +%s) < $(date -d "${end}" +%s) )); do
    echo "Processing date: $current_date"
    
    # 0-90-1 Step
    subdate=$current_date
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i1}" +%s) )); do
        echo "  - 0-90 Processing: $subdate until < ${current_date} ${i1}"
        subdate=$(date -d "${subdate} ${i2}")
        echo "    Updated subdate: $subdate"  # Output the updated subdate
    done
    echo "0-90 is done"
    subdate=$(date -d "${subdate} - 1 hour")
    subdate=$(date -d "${subdate} ${i22}")
    
    # 90-144-3 Step
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i11}" +%s) )); do
        echo "  - 90-144 Processing: $subdate until < ${current_date} ${i11}"
        subdate=$(date -d "${subdate} ${i22}")
        echo "    Updated subdate: $subdate"  # Output the updated subdate
    done
    echo "90-144 is done"
    subdate=$(date -d "${subdate} - 3 hours")
    #subdate=$(date -d "${subdate} ${i222}")

    # 144-240-6 Step
    while (( $(date -d "${subdate}" +%s) < $(date -d "${current_date} ${i111}" +%s) )); do
        echo "  - 144-240 Processing: $subdate until < ${current_date} ${i111}"
        subdate=$(date -d "${subdate} ${i222}")
        echo "    Updated subdate: $subdate"  # Output the updated subdate
    done
    echo "144-240 is done"
    
    echo "$current_date processed!"
    current_date=$(date -d "${current_date} ${i3}")
done

echo "Date processing simulation complete."
