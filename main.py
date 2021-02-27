import pandas as pd
columns = ["col1", "col2", "complexAge", "totalRooms", "totalBedrooms", "complexInhabitants", "apartmentsNr", "col8", "medianCompexValue"]
df = pd.read_csv("apartmentComplexData.txt", names=columns)
df.head()

X = df.iloc[:, 2:7]
Y = df.iloc[:,8]

#checking data for Nan values
df.isnull().sum()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#get constant and coeficient for linear regression
constant = lr.intercept_
coeficients = lr.coef_

print(f"Constant: {constant}")
print(f"Regression coeficients: {coeficients}")

y_pred = lr.predict(X_test)

df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_pred_head = df_pred.head(15)
df_pred_head
