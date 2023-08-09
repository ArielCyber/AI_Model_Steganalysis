import os

embedded = os.path.join('data', 'embedded')

features = os.path.join('data', 'features')
grads = os.path.join(features, 'grads')
losses = os.path.join(features, 'losses')

malware = os.path.join('data', 'malware')

model_zoos = os.path.join('data', 'model_zoos')

if '__main__' == __name__:
    # Create directories
    dirs = [embedded, features, grads, losses, malware, model_zoos]
    [os.makedirs(dir, exist_ok=True) for dir in dirs]

    zip_path = os.path.join(model_zoos, 'hyper_reps.zip')
    # Download model_zoos
    downlod_cmd = f'curl -# -o {zip_path} "https://zenodo.org/record/7529960/files/hyper_reps.zip?download=1"'
    old_cmd = f'wget -P {zip_path} "$https://zenodo.org/record/7529960/files/hyper_reps.zip?download=1"'
    if os.system(downlod_cmd):
        raise Exception('Failed to download model_zoos')

    os.system(f'unzip {zip_path} -d {model_zoos} && mv {model_zoos}/hyper_representations/* {model_zoos} && rm -rf {model_zoos}/hyper_representations && rm {zip_path}')