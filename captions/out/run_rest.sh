#!/bin/zsh
# Detached pipeline runner: precheck -> write -> validate -> repair -> review (single worker, resumable).
cd /Users/george_yang/workspace/code/wan_control_demo/captions
set -e
echo "=== $(date) start" >> out/pipeline.log
python3 -m capgen precheck --workers 1 >> out/pipeline.log 2>&1
python3 -m capgen write    --workers 1 >> out/pipeline.log 2>&1
python3 -m capgen validate --workers 1 >> out/pipeline.log 2>&1
python3 -m capgen repair   --workers 1 >> out/pipeline.log 2>&1
python3 -m capgen review               >> out/pipeline.log 2>&1
echo "=== $(date) DONE" >> out/pipeline.log
touch out/DONE
