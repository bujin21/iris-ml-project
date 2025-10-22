from src.data.loader import load_data
from src.features.preprocessing import preprocess
from src.models.model import IrisModel

# 데이터 로드
df = load_data()

# 전처리
X, y = preprocess(df)

# 모델 학습
model = IrisModel()
model.train(X, y)

# 예측
predictions = model.predict(X[:5])
print(predictions)
~                        