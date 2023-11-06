#!/usr/bin/env bash

############################################################
## Help                                                    #
############################################################

help() {
    echo "Transform training/val images from chosen directory into tfrecords with chosen number of images per record and image dimensions."
    echo
    echo "Syntax: tfrecords.sh [d|p|t|v|s]"
    echo "options:"
    echo "d     Directory containing training/val images. OPTIONAL"
    echo "p     Directory to save tfrecords. OPTIONAL"
    echo "t     Number of train images per tfrecord. OPTIONAL"
    echo "v     Number of val images per tfrecord. OPTIONAL"
    echo "s     Image height and width. REQUIRED"
    echo "m     Use multiprocessing. OPTIONAL"
    echo "h     Print this Help."

}

add_to_array_if_not_empty() {
    if [[ -z ${3+x} ]]; then
        echo "No $2 specified, using default."
    else
        echo "$2 found"
        array+=("-$1 $3")
    fi
}

############################################################

while getopts ':d:p:t:v:s:mh' opt; do
    case "$opt" in
    d)
        if [[ $OPTARG != '-p' ]]; then
            dir=$OPTARG
        fi
        ;;
    p)
        if [[ $OPTARG != '-t' ]]; then
            path=$OPTARG
        fi
        ;;
    t)
        if [[ $OPTARG != '-v' ]]; then
            train=$OPTARG
        fi
        ;;
    v)
        if [[ $OPTARG != '-s' ]]; then
            val=$OPTARG
        fi
        ;;
    s)
        size=($OPTARG)
        size=${size//,/ }
        ;;
    m)
        multi=$OPTARG
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
        exit 1
        ;;
    esac
done

array=("-s ${size[@]}")

add_to_array_if_not_empty d "Image directory" $dir
add_to_array_if_not_empty p "tfrecord path" $path
add_to_array_if_not_empty t "Number of train images per tfrecord" $train
add_to_array_if_not_empty v "Number of val images per tfrecord" $val

if [[ -z ${multi+x} ]]; then
    echo "No multiprocessing flag specified, using default."
    array+="-m"
else
    echo "multiprocessing flag found"
fi

# echo ${array[@]}
python src/data_processing/tfrecords.py ${array[@]}
