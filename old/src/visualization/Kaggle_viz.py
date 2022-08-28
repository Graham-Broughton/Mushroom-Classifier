def create_merged_df(images, category, annotations):
    """args = takes the three important df's and returns a single df"""
    
    data_df = images.copy()

    data_df['category_id'] = annotations[annotations['image_id'] == data_df['id']]['category_id']
    category.set_index('id')
    data_df = pd.merge(left=data_df, right=category, how='left', left_on='category_id', right_on='id')
    data_df.drop(columns='id_y', inplace=True)

    data_df['image_name'] = data_df['file_name'].str.split('/').str[2]
    data_df['subdir'] = data_df['file_name'].str.split('/',1).str[1]
    return data_df

def read_json(json_path):
    """ args = takes the Json files 
        returns = create the five dataframes for the five tables in the annotations json file
    """
    # # Opening JSON file
    f = open(json_path, )

    # returns JSON object as
    # a dictionary
    
    data = json.load(f)

    # Create different lists from the data dictionary

    annotations = pd.DataFrame(data["annotations"])
    category = pd.DataFrame(data["categories"])
    info = pd.DataFrame.from_dict(data["info"], orient='index')
    images = pd.DataFrame(data['images'])
    licenses = pd.DataFrame(data['licenses'])
    # Closing file
    f.close()
    return annotations, category, info, images, licenses

