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
    echo "y     Download images from FGVCX from your choice dataset. [2018|2019|2021|extras]"
    echo "a     Download images from all datasets"
    echo "h     Print this Help."
}

BASE="s3://ml-inat-competition-datasets"

dl_fgvcx_2018() {
    echo "Downloading the FGVCX 2018 data"
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2018/categories.json.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2018/inat2018_locations.zip ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2018/test2018.json.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2018/test2018.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2018/train2018.json.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2018/train_val2018.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2018/val2018.json.tar.gz ./data/raw/
}

dl_fgvcx_2019() {
    # Downloading the FGVCX 2021 data
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2019/train_val2019.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2019/test2019.json.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2019/val2019.json.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2019/categories.json.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2019/train2019.json.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2019/test2019.tar.gz ./data/raw/
}

dl_fgvcx_2021() {
    # Downloading the FGVCX 2021 data
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2021/train.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2021/train.json.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2021/val.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/2021/val.json.tar.gz ./data/raw/
}

dl_extras() {
    # Downloading the FGVCX extra data
    s5cmd --no-sign-request --numworkers 8 cp $BASE/inatloc/iNatLoc.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/newt/newt2021_images.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/newt/newt2021_labels.csv.tar.gz ./data/raw/
    s5cmd --no-sign-request --numworkers 8 cp $BASE/ssw60/ssw60.tar.gz ./data/raw/
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
        array=("2018" "2019" "2021" "extras")
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

echo "Number of arguments: ${#array[@]}"
echo -n "Arguments are:"
for i in "${array[@]}"; do
    echo -n " ${i},"
done
echo ''
############################################################

echo "Downloading images from FGVCX.."
for i in "${array[@]}"; do
    if [ "${i}" = "2018" ]; then
        dl_fgvcx_2018
    elif [ "${i}" = "2019" ]; then
        dl_fgvcx_2019
    elif [ "${i}" = "2021" ]; then
        dl_fgvcx_2021
    elif [ "${i}" = "extras" ]; then
        dl_extras
    fi
done

############################################################