import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_DATA_DIR

# Load processed data
data = pd.read_csv(PROCESSED_DATA_DIR / "dataset.csv")
print(f"Training data length: {len(data)}")

y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)