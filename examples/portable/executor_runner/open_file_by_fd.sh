#!/bin/bash
FILENAME="$1"
exec {FD}<${FILENAME}     # open file for read, assign descriptor
echo "Opened ${FILENAME} for read using descriptor ${FD}"
out/host/linux-x86/bin/executor_runner --model_path fd:///${FD} --is_fd_uri=true
exec {FD}<&-    # close file
