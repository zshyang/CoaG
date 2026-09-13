#!/bin/bash
# Pack the staged training set (hard links) into one uncompressed tar for transfer (~9.2 GB).
set -e
cd /Users/george_yang/workspace/runtime/wan_control_demo
tar -cf train_data.tar train_data && ls -la train_data.tar | awk '{print $5, $9}'
