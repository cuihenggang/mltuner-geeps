#!/usr/bin/env sh

python scripts/summarize-accuracy.py $1/output.txt > $1/accuracy.txt
python scripts/summarize-accuracy2.py $1/output.txt > $1/accuracy2.txt
python scripts/summarize-retune.py $1/output.txt > $1/retunes.txt
python scripts/summarize-times.py $1/output.txt > $1/times.txt
python scripts/prune-log.py $1/output.txt > $1/output.pruned.txt
