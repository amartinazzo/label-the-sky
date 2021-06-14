import yaml

CLASS_MAP = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)