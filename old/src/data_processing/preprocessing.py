import os
import imghdr
import pandas as pd
import regex as re
import pickle

def rename_images(fpath, label):
    """
    renames the files for a given directory
    """
    for num, fname in enumerate(os.listdir(fpath)[1:-2]):
        new_name = re.sub('\.\d+', '', fname)
        new_name = str(num)+new_name
        new_name = new_name + label
        os.rename(os.path.join(fpath, fname), os.path.join(fpath, new_name))

def return__df(filepath, label):
    """
    takes a GCS directory and returns a df with the filepath of everything in the gcloud directory
    with a column containging a given label
    """
    dir_list = !gsutil ls filepath
    df = pd.DataFrame(dir_list)[1:, :]
    df['label'] = label
    return df

def fix_filename(size, filepath=filepath, df=df):
    for fname in os.listdir(filepath):
        new = re.sub('({size}\.\d?\.)(\D?)(\.\d)', r'\2\1', fname)
        new = label+new
        os.rename(os.path.join(filepath, fname), os.path.join(filepath, new))
    df[0] = df[0].apply(lambda x: re.sub('({size}.png)\.(\d)', r'\2\1', x))
    return df

def split_file_name(df):
    """ splits a df on / and selects the last item, good for grabbing just the file name
    """
    df1['split'] = df1['file_name'].apply(lambda x: x.split('/')[-1])
    return df

def to_content(df1, df2):
    """
    takes two dfs (train and val) and replaces the GCS prefix with the local dir
    """
    df1['file_name'] = df1['file_name'].replace('gs://medium_mush', '/content/mush', regex=True)
    df2['file_name'] = df2['file_name'].replace('gs://medium_mush', '/content/mush', regex=True)
    return df1, df2


def break_dfs_smaller(train, val, lower_limit, upper_limit):
    """
    takes two dfs (train and val) and selects for category_ids counts above the lower limit
    then selects the top upper limit of category ids.
    So it selects the range of counts for category ID you want.
    """
    counts = df1.groupby(['category_id', 'name']).count().reset_index()
    counts = counts[counts['id_x'] > lower_limit]
    df1 = df1[df1['name'].isin(counts['name'])]
    df2 = df2[df2['name'].isin(counts['name'])]
    df1.groupby('category_id').head(upper_limit)
    return df1, df2

def remove_corrupted_images(path, dir):
    """
    check for and remove corrupted images in given directory
    """
    for image in dir:
        file = os.path.join(path, image)
        if not imghdr.what(file):
            print(file)
            os.remove(file)

def save_list(list_, path):
    with open(path, 'wb') as f:
        pickle.dump(list_, f)

def load_list(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
