
from glob import glob
import os
from pathlib import Path
import pandas as pd

def get_all_images_from_dir(cur_dir,ext="png"):
	"""
	List out all the images which exists under the passed cur_dir
	cur_dir = the directory under which you want to find images
	ext = The extension for the files that you want to search
	returns : list of image urls 
	"""

	allowed_image_ext = {'jpg','jpeg','png'}

	if ext not in allowed_image_ext:
		print("The passed ext is not supported . Supported extensions are ... ", allowed_image_ext)
		return None
	paths = list(map(str,Path(cur_dir).glob('**/*.'+ext)))
	return paths

def urls_to_df(url_list):
	return pd.DataFrame(url_list,columns=['file_location'])

def if_path_exists(target_file):
	return Path(target_file).exists()

def create_subdirs(file_path,overwrite=False):
	"""
	Creates sub dir under a given path
	"""

	
	actual_path = get_files_parent(file_path)

	if if_path_exists(actual_path) : 
		print("File already exists ... ")
		if not overwrite :
			print("exiting ...")
			return None
		print("overwriting ...")
		actual_path.mkdir(exist_ok=True,parents=True)
		return True

	print("Creating directory ")
	actual_path.mkdir(exist_ok=True,parents=True)
	return True

def get_files_parent(file_path):
	file_url = Path(file_path)
	# if not file_url.is_file():
	# 	print("This function only returns file's parent")
	# 	return 0
	return Path(file_path).parent

def get_stem(file_path):
	return Path(file_path).stem




def main():
	#print(urls_to_df(get_all_images_from_dir('/home/ubuntu/Documents/dataset/nsfw',ext='jpg')))
	create_subdirs('abhay','ram/rahim')


if __name__=="__main__":
	import sys
	sys.exit(main())