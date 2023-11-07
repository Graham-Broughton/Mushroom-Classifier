#!/usr/bin/env bash

BASE="s3://ml-inat-competition-datasets"
TARGET_DIR="./training/data/raw"

TRAIN_IMGS_2021_MD5="e0526d53c7f7b2e3167b2b43bb2690ed"
TRAIN_IMGS_MINI_2021_MD5="db6ed8330e634445efc8fec83ae81442"
TRAIN_IMGS_2018_MD5="b1c6952ce38f31868cc50ea72d066cc3"


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
    echo "Checking MD5 sums"
    if [ "$YEAR" == "2021" ]; then
        echo "$TRAIN_IMGS_2021_MD5 $TARGET_DIR/$YEAR/train.tar.gz" | md5sum -c
        echo "$TRAIN_IMGS_MINI_2021_MD5 $TARGET_DIR/$YEAR/train_mini.tar.gz" | md5sum -c
    elif [ "$YEAR" == "2018" ]; then
        echo "$TRAIN_IMGS_2018_MD5 $TARGET_DIR/$YEAR/train_val2018.tar.gz" | md5sum -c
    fi
    # echo "Extracting & Deleting the FGVCX $YEAR data"
}

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

for i in "${array[@]}"; do
    YEAR=$i dl_fgvcx_year
done

############################################################