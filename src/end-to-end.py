import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

test=pd.read_csv("house-prices-advanced-regression-techniques/test.csv")
sample_submission=pd.read_csv("house-prices-advanced-regression-techniques/sample_submission.csv")

#load the old data to creat our model 
df=pd.read_csv("house-prices-advanced-regression-techniques/train.csv")

#EDA 
# print(df.info())
# print(df.head())
# print(df.tail())
# print(df.describe())
# print(df.shape)
# print(df['SalePrice'].head())

#cleaning data
# 1. Missing values
df.isnull().sum().sort_values(ascending=False)

# 2. Drop highly missing columns
df.drop(columns=['PoolQC', 'Alley', 'Fence'], inplace=True)

# 3. Fill categorical missing values
for col in df.select_dtypes(include=['object', 'string']):
    df[col] = df[col].fillna('None')

# 4. Fill numerical missing values
for col in df.select_dtypes(include=['int64', 'float64']):
    df[col] = df[col].fillna(df[col].median())

#convert text into numbers 

df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

# One-hot encoding for categorical features
df = pd.get_dummies(df, drop_first=True)

#ML model
##tain/test split
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#train the model 
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


#cleaning test data 
# 1. Missing values
test.isnull().sum().sort_values(ascending=False)

# 2. Drop highly missing columns
test.drop(columns=['PoolQC', 'Alley', 'Fence'], inplace=True)

# 3. Fill categorical missing values
for col in test.select_dtypes(include=['object', 'string']):
    test[col] = test[col].fillna('None')

# 4. Fill numerical missing values
for col in test.select_dtypes(include=['int64', 'float64']):
    test[col] = test[col].fillna(test[col].median())
    
#convert text into numbers 

test= pd.get_dummies(test, columns=['Neighborhood'], drop_first=True)

# One-hot encoding for categorical features
test= pd.get_dummies(test, drop_first=True)
test = test.reindex(columns=X.columns, fill_value=0)

print(X.shape)
print(test.shape)
print(test.info())


#make predictions
predictions = model.predict(test)
predictions = np.maximum(predictions, 0)

print(predictions[:10])

output = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": predictions
})

output.to_csv("submission.csv", index=False)
print(output.head())



#visualization
##Actual vs Predicted Prices

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.savefig("Actual vs Predicted Prices")


##Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle='--')

plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.savefig("Residual Plot")
