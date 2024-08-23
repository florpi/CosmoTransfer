#!/bin/bash

SOURCE_ENDPOINT="e0eae0aa-5bca-11ea-9683-0e56c063f437"
DESTINATION_ENDPOINT="YOUR-END-POINT-ID"
SOURCE_PATH="/3D_cubes/BSQ/"
DESTINATION_PATH="YOUR-DESTINATION-PATH"

TRANSFER_LIST="transfer_list.txt"

# Clear the previous transfer list if it exists
> $TRANSFER_LIST

echo "Listing directories in $SOURCE_PATH on Globus..."
dirs=$(globus ls $SOURCE_ENDPOINT:$SOURCE_PATH) 
dir_count=$(echo "$dirs" | wc -l)
echo "Number of directories found in $SOURCE_PATH: $dir_count"

# Loop over each directory found and prepare the transfer list
for i in $dirs; do
    echo "Adding $i to transfer list"
    echo "$SOURCE_PATH/$i/df_m_CIC_z=0.00.hdf5 $DESTINATION_PATH/$i/df_m_CIC_z=0.00.hdf5" >> $TRANSFER_LIST
done

# Start the Globus transfer
echo "Initiating Globus transfer..."
task_output=$(globus transfer $SOURCE_ENDPOINT $DESTINATION_ENDPOINT --batch $TRANSFER_LIST)
task_id=$(echo "$task_output" | grep 'Task ID' | awk '{print $3}')

# Check if the task ID was successfully captured
if [ -z "$task_id" ]; then
    echo "Error: Task ID not captured. Here is the output of the globus transfer command:"
    echo "$task_output"
    exit 1
else
    echo "Task ID: $task_id"
fi

# Periodically check the transfer status
while true; do
    status=$(globus task show $task_id | grep 'Status' | awk '{print $2}')
    echo "Current transfer status: $status"
    if [ "$status" == "SUCCEEDED" ]; then
        echo "Transfer completed successfully."
        break
    elif [ "$status" == "FAILED" ]; then
        echo "Transfer failed."
        exit 1
    else
        echo "Transfer is still in progress... (Status: $status)"
    fi
    sleep 60  # Wait for 60 seconds before checking again
done

# Conversion section
for i in $dirs; do
    HDF5_FILE="$DESTINATION_PATH/$i/df_m_CIC_z=0.00.hdf5"
    NUMPY_FILE="$DESTINATION_PATH/$i/df_m_CIC_z=0.00.npy"

    echo "Processing directory: $i"
    echo "Checking if file exists: $HDF5_FILE"

    if [ -f $HDF5_FILE ]; then
        echo "File $HDF5_FILE found, converting to NumPy..."
        python3 -c "
import numpy as np
import h5py

try:
    print('Opening file: $HDF5_FILE')
    with h5py.File('$HDF5_FILE', 'r') as f:
        print('Reading dataset: df')
        data = f['df'][:]
    print('Saving NumPy file: $NUMPY_FILE')
    np.save('$NUMPY_FILE', data)
    print(f'Converted {HDF5_FILE} to $NUMPY_FILE')
except KeyError:
    print(f'Dataset name not found in {HDF5_FILE}. Please check the dataset name.')
except Exception as e:
    print(f'Error processing $HDF5_FILE: {e}')
"
        if [ -f $NUMPY_FILE ]; then
            echo "Conversion successful, deleting $HDF5_FILE..."
            rm $HDF5_FILE
            echo "Deleted $HDF5_FILE after conversion."
        else
            echo "Conversion failed, not deleting $HDF5_FILE."
        fi
    else
        echo "File $HDF5_FILE does not exist, skipping..."
    fi
done

echo "Script execution completed."
