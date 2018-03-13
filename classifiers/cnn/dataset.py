import pandas as pd


def get_corpus(file_path):
    df = pd.read_csv(file_path)

    y = df.label.to_matrix()
