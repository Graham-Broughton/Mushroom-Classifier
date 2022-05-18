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

    