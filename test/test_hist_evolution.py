import numpy as np

def calculate_skewness_kurtosis(data):
    """
    Calcola il coefficiente di asimmetria e la curtosi di una distribuzione
    :param data: array contenente i dati della distribuzione
    :return: coeffiente di asimmetria e curtosi
    """
    mean = np.mean(data)
    std = np.std(data)
    skew = (np.mean((data - mean) ** 3) / std ** 3)
    kurt = (np.mean((data - mean) ** 4) / std ** 4) - 3
    return skew, kurt

print(calculate_skewness_kurtosis([1, 1, 1, 1, 1.1]))