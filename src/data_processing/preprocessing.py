
def rename_images(fpath, label):
    for num, fname in enumerate(os.listdir(fpath)[1:-2]):
        new_name = re.sub('\.\d+', '', fname)
        new_name = str(num)+new_name
        new_name = new_name + label
        os.rename(os.path.join(fpath, fname), os.path.join(fpath, new_name))

def return_labeled_df(filepath, cloud_path, label):
    dir_list = []
    for fname in os.listdir(filepath):
        new = fname + ',' + label
        new = cloud_path + new
        dir_list.append(new)
    return pd.DataFrame(dir_list)

def fix_filename(size, filepath=filepath, df=df):
    for fname in os.listdir(filepath):
        new = re.sub('({size}\.\d?\.)(\D?)(\.\d)', r'\2\1', fname)
        new = label+new
        os.rename(os.path.join(filepath, fname), os.path.join(filepath, new))
    df[0] = df[0].apply(lambda x: re.sub('({size}.png)\.(\d)', r'\2\1', x))
    return df

def csv_to_txt(f1, f2, f3, f4, source, dest):
	"""
	"""
	asco = pd.read_csv(source + f1 + '.csv', header=None)[0].values.tolist()
	basidio = pd.read_csv(source + f2 + '.csv', header=None)[0].values.tolist()
	gyro = pd.read_csv(source + f3 + '.csv', header=None)[0].values.tolist()
	morel = pd.read_csv(source + f4 + '.csv', header=None)[0].values.tolist()

	with open(dest+f1+'.txt', 'w') as f:
		for item in asco:
        	f.write(f'{item}\n')

	with open(dest+f2+'.txt', 'w') as f:
	    for item in nb:
        	f.write('%s\n' % item)

	with open(dest+f3+'.txt', 'w') as f:
	    for item in gyro:
	        f.write(f'{item}\n')

	with open(dest+f4+'.txt', 'w') as f:
	    for item in morel:
	        f.write(f'{item}\n')
    return

