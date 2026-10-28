#!/bin/sh
cd /Users/hemduttrao/adr_longshort
while pgrep -f "p2_buildup.py buildup" >/dev/null; do sleep 20; done
.venv/bin/python research/p2_buildup.py pairs > research/out/p2_log_pairs.txt 2>&1
.venv/bin/python research/p2_confirm.py > research/out/p2_log_confirm.txt 2>&1
