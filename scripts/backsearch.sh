#! /bin/bash
if [ "$#" -lt 1 ] ; then
    echo "Usage: $0 [nstep]" >&2
    exit 1
fi
nstep=$1
shift
python src/main.py --cuda-use --checkpoint-dir-name "backsearch-$nstep" --mode 0 --teacher-forcing-ratio 0 --fix-rng --use-rule --run-flag=backsearch --nstep=$nstep $@
