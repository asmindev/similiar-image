import os


for dirname, _, filenames in os.walk('static/img/dataset-image/saleh/'):
    for index, filename in enumerate(filenames):
        last_name = filename.split('/')[-1]
        rename = filename.replace(last_name, f"{index}.jpg")
        print(f"Renaming {filename} to {rename}")
        os.rename(os.path.join(dirname, filename), os.path.join(dirname, rename))
