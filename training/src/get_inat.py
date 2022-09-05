import pandas as pd
import json
import wget
import tarfile
import os
import tensorflow as tf
from tensorflow.keras import applications

def read_jsons(filepath):
    """
    Reads the unzipped tar json files and saves the information from each key in a data frame
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    try:
        annotations = pd.DataFrame(data['annotations'])
        category = pd.DataFrame(data['categories'])
        info = pd.DataFrame.from_dict(data['info'], orient='index')
        images = pd.DataFrame(data['images'])
        licenses = pd.DataFrame(data['licenses'])
        return annotations, category, info, images, licenses

    except KeyError:
        info = pd.DataFrame.from_dict(data['info'], orient='index')
        images = pd.DataFrame(data['images'])
        licenses = pd.DataFrame(data['licenses'])
        return info, images, licenses

def read_aws_jsons(filepath):

    annot, cat, info, images, license = read_jsons(filepath)

    df = merge_dfs(images, annot, cat)

    df = df[df['kingdom'] == 'Fungi']
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['day'] = pd.to_datetime(df['date']).dt.dayofyear
    df['date'] = pd.to_datetime(df['date']).dt.date

    df.drop(['year', 'image_dir_name', 'kingdom', 'phylum', 'class', 'order', 'family', 'common_name', 'supercategory', 'location_uncertainty'], axis=1, inplace=True)
    df.set_index('id_x', inplace=True)
    return df

def remove_images_not_fungi(name, df):
    df_list = df['sub_dir'].values.tolist()
    dirs = os.listdir(name)
    remove_list = [x for x in dirs if x not in df_list]

    for path in remove_list:
        shutil.rmtree(name+'/'+path)

    for path in os.listdir(name):
        split = path.split('_')
        num = split[0]
        newname = split[-2:]
        newname.insert(0, num)
        new = '_'.join(newname)
        os.rename(name+'/'+path, name+'/'+new)

def create_labels(train, val):
    cat_names = val['name'].unique().tolist()

    categories = pd.DataFrame(cat_names)
    categories.reset_index(inplace=True)
    categories.columns = ['label', 'name']

    categories.set_index('name', inplace=True)
    val.set_index('name', inplace=True)
    train.set_index('name', inplace=True)

    val = val.merge(categories, on='name', how='left')
    train = train.merge(categories, on='name', how='left')

    train.reset_index(inplace=True)
    val.reset_index(inplace=True)
    return train, val

def create_filepaths(train, val):
    train['sub_dir'] = train.apply(lambda x: "_".join((x['sub_dir'].split('_')[0], x['genus'], x['specific_epithet'])), axis=1)
    val['sub_dir'] = val.apply(lambda x: "_".join((x['sub_dir'].split('_')[0], x['genus'], x['specific_epithet'])), axis=1)

    val['filepath'] = 'val/'+val['sub_dir']+'/'+val['image_name']
    train['filepath'] = 'train/'+train['sub_dir']+'/'+train['image_name']
    return train, val