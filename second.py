import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

file_path = 'train.csv'  
data = pd.read_csv(file_path)

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

filtered_data = data[features + [target]].dropna()

X = filtered_data[features]
y = filtered_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)


sample_input = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2]
})
sample_prediction = model.predict(sample_input)
print("Predicted price for sample input:", sample_prediction[0])
