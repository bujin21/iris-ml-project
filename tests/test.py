from src.data.loader 

import load_data from src.features.preprocessing 

import preprocess from src.models.model 

import IrisModel 


df = load_data()                      # 데이터 로드

X, y = preprocess(df)                 # 전처리


model = IrisModel()                   # 모델 학습

model.train(X, y)


predictions = model.predict(X[:5])    # 예측

print(predictions)
