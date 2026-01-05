import pandas as pd

def load_csv(file, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """
    CSV'yi DataFrame'e çevirir.
    Bu fonksiyonun amacı: streamlit UI'dan bağımsız, test edilebilir IO katmanı yaratmak.
    """
    return pd.read_csv(file, sep=sep, encoding=encoding)
