#!/usr/bin/env bash
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

BASE="s3://ml-inat-competition-datasets"

dl_fgvcx_2018() {
    echo "Downloading the FGVCX 2018 data"
    s5cmd --no-sign-request cp --concurrency 20 $BASE/2018/* ./data/raw/2018/
}

dl_fgvcx_2019() {
    echo "Downloading the FGVCX 2019 data"
    s5cmd --no-sign-request cp --sp --concurrency 20 $BASE/2019/* ./data/raw/2019/
}

dl_fgvcx_2021() {
    echo "Downloading the FGVCX 2021 data"
    s5cmd --no-sign-request cp --sp --concurrency 20 $BASE/2021/* ./data/raw/2021/
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

echo "Downloading images from FGVCX.."
for i in "${array[@]}"; do
    if [ "${i}" = "2018" ]; then
        dl_fgvcx_2018
    elif [ "${i}" = "2019" ]; then
        dl_fgvcx_2019
    elif [ "${i}" = "2021" ]; then
        dl_fgvcx_2021
    fi
done

############################################################