import yaml

BANDS = ['u','f378','f395','f410','f430','g','f515','r','f660','i','f861','z']
CLASS_MAP = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}

config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)