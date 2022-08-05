import pandas as pd
import json
import wget
import tarfile
import os

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

def merge_dfs(images, annotation, category):
    """
    takes the three useful dataframes: images, annotations and categories, and merges them into one
    """
    df = images.copy()

    df['category_id'] = annotation[annotation['image_id'] == df['id']]['category_id']

    category.set_index('id')
    df = pd.merge(left=df, right=category, how='left', left_on='category_id', right_on='id')

    df['image_name'] = df['file_name'].str.split('/').str[-1]
    df['sub_dir'] = df['file_name'].str.split('/').str[-2]

    df.drop(['id_y', 'license', 'rights_holder', 'file_name'], axis=1, inplace=True)
    return df


train_URL = 'https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz'
test_URL = 'https://labs.gbif.org/fgvcx/2018/fungi_test.tgz'
annotation_URL = 'https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz'
test_info_URL = 'https://raw.githubusercontent.com/visipedia/fgvcx_fungi_comp/master/data/test_information.tgz'

wget.download(train_URL)
wget.download(test_URL)
wget.download(annotation_URL)
wget.download(test_info_URL)

with tarfile.open('train_val_annotations.tgz') as f:
    f.extractall()
with tarfile.open('fungi_test.tgz') as f:
    f.extractall()
with tarfile.open('fungi_train_val.tgz') as f:
    f.extractall()
with tarfile.open('test_information.tgz') as f:
    f.extractall()

tr_annotation, tr_category, tr_info, tr_images, tr_licenses = read_jsons('train.json')
val_annotation, val_category, val_info, val_images, val_licenses = read_jsons('val.json')
test_info, test_images, test_licenses = read_jsons('test.json')

train_annot = merge_dfs(tr_images, tr_annotation, tr_category)
val_annot = merge_dfs(val_images, val_annotation, val_category)

test_images['new_filename'] = 'test/' + test_images['id'].astype(str) + '.jpg'

test_images.to_csv('test.csv', index=False)
train_annot.to_csv('train.csv', index=False)
val_annot.to_csv('val.csv', index=False)

for filename in os.listdir('test'):
    try:
        new_file = test_images[test_images['file_name'].str.split('/').str[-1] == filename]['new_filename'].values[0]
        os.rename('test/'+filename, new_file)
    except IndexError:
        continue