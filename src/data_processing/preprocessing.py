import json
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np
from prefect import flow, task

from loguru import logger

logger.remove(0)


@task
def class_priors(df):
    class_priors = np.zeros(len(df['class_id'].unique()))
    for species in df['class_id'].unique():
        class_priors[species] = len(df[df['class_id'] == species])

    class_priors = class_priors/sum(class_priors)
    return class_priors


@task
def month_distributions(df):
    month_distributions = {}

    for _, observation in tqdm(df.iterrows(), total=len(df)):
        month = str(observation.month)
        if month not in month_distributions:        
            month_distributions[month] = np.zeros(len(df['class_id'].unique()))
        else:
            class_id = observation.class_id
            month_distributions[month][class_id] += 1

    for key, value in month_distributions.items():
        month_distributions[key] = value / sum(value)
    return month_distributions


@task
def parse_json(filepath, is_test=False, categories=None):
    with open(filepath, 'r') as f:
        res = json.load(f)
    info = pd.DataFrame.from_dict(res['info'], orient='index')
    images = pd.DataFrame(res['images']).set_index('id')
    if not is_test:
        annotations = pd.DataFrame(res['annotations']).set_index('id')
        if categories:
            categories = pd.DataFrame(res['categories']).set_index('id')
            return info, images, annotations, categories
        return info, images, annotations

    return info, images


@task
def join_dataframes(images, annotations, categories, locations=None):
    categories = categories[categories['supercategory'] == 'Fungi']
    categories = categories.rename(columns={'id': 'category_id'})
    if locations is None:
        df = categories.merge(annotations, right_on='category_id', left_index=True, how='inner').merge(
            images, left_on='image_id', right_index=True, how='inner'
        )
    else:
        df = (
            categories.merge(annotations, right_on='category_id', left_index=True, how='inner')
            .merge(images, left_on='image_id', right_index=True, how='inner')
            .merge(locations, left_on='image_id', right_index=True, how='inner')
        )
    try:
        df = df.drop(['supercategory', 'kingdom', 'image_id', 'valid', 'license', 'rights_holder', 'user_id'], axis=1)
    except KeyError:
        df = df.drop(['supercategory', 'kingdom', 'image_id', 'license', 'rights_holder'], axis=1)
    finally:
        return df


@flow(name='Parse2018Data')
def parse_2018_data(data_root):
    logger.debug(f'Parsing 2018 data from {data_root}')
    _, timages2018, tanno2018 = parse_json(data_root / 'train2018.json')
    _, vimages2018, vanno2018 = parse_json(data_root / 'val2018.json')

    tloc = pd.read_json(data_root / 'inat2018_locations' / 'train2018_locations.json').set_index('id')
    vloc = pd.read_json(data_root / 'inat2018_locations' / 'val2018_locations.json').set_index('id')

    with open(data_root / 'categories.json', 'r') as f:
        res = json.load(f)
        cats = pd.DataFrame(res)

    val = join_dataframes(vimages2018, vanno2018, cats, vloc)
    val['set'] = "val"

    train = join_dataframes(timages2018, tanno2018, cats, tloc)
    train['set'] = 'train'
    df = pd.concat([train, val]).reset_index(drop=True)

    df['name'] = df['name'].apply(lambda x: '_'.join(x.split(' ')))
    df['new_dirs'] = df.apply(lambda x: f"Fungi_{x['phylum']}_{x['class']}_{x['order']}_{x['family']}_{x['name']}", axis=1)
    df['new_paths'] = df['new_dirs'] + '/' + df['file_name'].str.split('/').str[-1]

    df.loc[:, 'file_name'] = df['file_name'].str.split('/').str[-1]
    df = df.drop(['category_id', 'new_paths'], axis=1)
    df['specific_epithet'] = df['name'].str.split(' ').str[-1]
    df['image_dir_name'] = (
        "Fungi_"
        + df['phylum']
        + '_'
        + df['class']
        + '_'
        + df['order']
        + '_'
        + df['family']
        + '_'
        + df['genus']
        + '_'
        + df['specific_epithet']
    )
    df = df.rename(columns={'lon': 'longitude', 'lat': 'latitude', 'loc_uncert': 'location_uncertainty'})

    df['dataset'] = '2018'
    df['date'] = pd.to_datetime(df['date'])
    return df


@flow(name='Parse2021Data')
def parse_2021_data(data_root):
    _, timages2021, tanno2021, tcat2021 = parse_json(data_root / 'train.json', categories=True)
    _, vimages2021, vanno2021, vcat2021 = parse_json(data_root / 'val.json', categories=True)

    train = join_dataframes(timages2021, tanno2021, tcat2021)
    val = join_dataframes(vimages2021, vanno2021, vcat2021)
    train['set'] = 'train'
    val['set'] = 'val'
    df = pd.concat([train, val]).reset_index(drop=True)

    df['image_dir_name'] = df['image_dir_name'].apply(lambda x: '_'.join(x.split('_')[1:]))
    df = df.drop('category_id', axis=1)
    df['dataset'] = '2021'
    df['file_name'] = df['file_name'].str.split('/').str[-1]
    df['date'] = pd.to_datetime(df['date'])
    return df


@flow(name='JoinDatasets')
def join_datasets(gcs_bucket, root):
    df1 = parse_2018_data(root / '2018')
    df2 = parse_2021_data(root / '2021')
    df = pd.concat([df1, df2], ignore_index=True)
    
    df['file_path'] = 'Mushroom-Classifier/data/train/' + df['image_dir_name'] + '/' + df['file_name']
    df['gcs_path'] = f'gs://{gcs_bucket}/train/' + df['image_dir_name'] + '/' + df['file_name']
    df['class_id'] = df.groupby('name').ngroup()

    month_distribution = month_distributions(df)
    class_prior = class_priors(df)
    return df, month_distribution, class_prior


if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    root = Path('data')
    df, month_distribution, class_prior = join_datasets(os.environ['GCS_BUCKET'], root)
    df.to_csv(root / 'train_val.csv', index=False)
