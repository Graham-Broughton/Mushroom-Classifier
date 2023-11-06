#!/usr/bin/env bash

BASE="s3://ml-inat-competition-datasets"
TARGET_DIR="./training/data/raw"

############################################################
# Help                                                     #
############################################################

help() {
    # Display Help
    echo "Download images for mushroom classification from your choice of datasets."
    echo
    echo "Syntax: get_data.sh [y|a|h]"
    echo "options:"
    echo "y     Download images from FGVCX from your choice of year. [2018|2019|2021]"
    echo "a     Download images from all datasets"
    echo "h     Print this Help."
}

############################################################
# Functions                                               #
############################################################

dl_fgvcx_year() {
    echo "Downloading the FGVCX $YEAR data"
    s5cmd --no-sign-request cp --concurrency 20 $BASE/$YEAR/* $TARGET_DIR/$YEAR/
}

# dl_fgvcx_2019() {
#     echo "Downloading the FGVCX 2019 data"
#     s5cmd --no-sign-request cp --sp --concurrency 20 $BASE/2019/* $TARGET_DIR/2019/
# }

# dl_fgvcx_2021() {
#     echo "Downloading the FGVCX 2021 data"
#     s5cmd --no-sign-request cp --sp --concurrency 20 $BASE/2021/* $TARGET_DIR/2021/
# }

############################################################

while getopts ':y:ah' opt; do
    case "$opt" in
    y)
        set -f
        IFS=' '
        array=($OPTARG)
        ;;
    a)
        set -f
        IFS=' '
        array=("2018" "2019" "2021")
        ;;
    h)
        help
        exit
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1
        ;;
    *)
        echo "Incorrect input, please review the following help menu."
        help
        exit
        ;;
    esac
done

############################################################

# echo "Downloading images from FGVCX.."
# for i in "${array[@]}"; do
#     if [ "${i}" = "2018" ]; then
#         dl_fgvcx_2018
#     elif [ "${i}" = "2019" ]; then
#         dl_fgvcx_2019
#     elif [ "${i}" = "2021" ]; then
#         dl_fgvcx_2021
#     fi
# done

echo "Downloading images from FGVCX.."
for i in "${array[@]}"; do
    YEAR=$i dl_fgvcx_year
done

############################################################