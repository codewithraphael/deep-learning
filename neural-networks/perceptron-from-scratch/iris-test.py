import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from src.perceptron import Perceptron

from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / 'data'
PLOTS_DIR = PROJECT_ROOT / 'plots'
MODEL_DIR = PROJECT_ROOT / 'models'

DATA_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================
#  LOADING DATASET
# =========================
def load_data(filepath):
    data = pd.read_csv(filepath, header=None, encoding='utf-8')

    return data


# =========================
#  EDA
# =========================
def eda(data):
    print(f'\n ===== Shape of the dataset ===== \n {data.shape}')
    print(data.info())
    print(f'\n ===== Summary Statistics ===== \n {data.describe()}')
    print(f'\n ===== Shape of the dataset ===== \n {data.shape}')
    





# =========================
#  MAIN
# =========================
def main():
    filepath = PROJECT_ROOT / 'data/iris.csv'
    data = load_data(filepath)
    eda(data)







if __name__ == '__main__':
    main()



