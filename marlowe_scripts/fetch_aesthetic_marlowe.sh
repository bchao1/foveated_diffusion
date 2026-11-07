#!/bin/bash
# Download + extract Aesthetic-Train-V2 (HF rev 3902ab5) into $DST/images/NNNN/. Safe to rerun.
DST=/scratch/m000051-pm07/brianchc/datasets/Aesthetic-Train-V2
REV=3902ab5cc528dbcb5de8e20d41e845c68b48a66a
cd "$DST" || exit 1; mkdir -p images _tars
one() {
    local id=$1 size=$2
    [ -f "images/$id/.done" ] && return 0
    local t="_tars/$id.tar"
    for try in 1 2 3; do
        curl -sfL -C - -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" -o "$t" \
          "https://huggingface.co/datasets/zhang0jhon/Aesthetic-Train-V2/resolve/$REV/images/$id.tar"
        [ "$(stat -c %s "$t" 2>/dev/null)" = "$size" ] && break
        echo "$id: size mismatch (try $try)"; rm -f "$t"
    done
    [ "$(stat -c %s "$t" 2>/dev/null)" = "$size" ] || { echo "$id: FAILED"; return 1; }
    tar xf "$t" -C images && touch "images/$id/.done" && rm -f "$t" && echo "$id ok $(date +%T)"
}
export -f one; export REV
xargs -P 3 -n 2 bash -c 'one "$@"' _ < tar_sizes.txt
echo "ALL DONE $(date -Is): $(ls images/*/.done | wc -l)/106 tars"
