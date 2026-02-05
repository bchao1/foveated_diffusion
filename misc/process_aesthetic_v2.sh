#!/bin/bash

image_dir="/miele/brian/data/Aesthetic-Train-V2/images"
output_dir="/local/brian/Aesthetic-Train-V2/images"

# Number of parallel jobs
MAX_JOBS=8

cd "$image_dir"

# Extract each tar file (0000.tar to 0105.tar) to output_dir/
# Note: tar files already contain ABCD/ directory structure inside
mkdir -p "${output_dir}"

job_count=0
for tar_file in $(seq -w 0000 0105); do
    if [ -f "${tar_file}.tar" ]; then
        echo "Extracting ${tar_file}.tar to ${output_dir}/"
        tar -xf "${tar_file}.tar" -C "${output_dir}" &
        
        ((job_count++))
        
        # Wait if we've hit max parallel jobs
        if [ $job_count -ge $MAX_JOBS ]; then
            wait
            job_count=0
        fi
    else
        echo "Warning: ${tar_file}.tar not found, skipping..."
    fi
done

# Wait for remaining jobs
wait

echo "Done extracting all tar files."