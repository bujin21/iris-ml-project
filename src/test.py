from src.data.loader import load_data
from src.features.preprocessing import preprocess
from src.models.model import IrisModel

# ������ �ε�
df = load_data()

# ��ó��
X, y = preprocess(df)

# �� �н�
model = IrisModel()
model.train(X, y)

# ����
predictions = model.predict(X[:5])
print(predictions)
~                        