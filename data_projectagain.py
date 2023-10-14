import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from scipy.stats import linregress
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Download the CSV file
url = "https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv"
response = requests.get(url)

with open('50_Startups.csv', 'wb') as f:
    f.write(response.content)

# Read the CSV file into a DataFrame
df = pd.read_csv('50_Startups.csv')

# statistical data
column_stats = df.describe()
# print(column_stats)

# look for duplicates

for column in df.columns:
    duplicates = df[column].duplicated()

    if duplicates.any():
        print(f"Column '{column} has {duplicates.sum()} duplicates")
        print(df[duplicates][column])
    else:
        print(f"No missing values found in column '{column}'")
# look for empty cells
for column in df.columns:
    # check for missing values in current column
    missing_values = df[column].isna() # returns a boolean series where 'True' means a missing value
    # Print column name and missing values (if any)
    if missing_values.any():
        print(f"Column '{column}' has {missing_values.sum()} missing values:")
        print(df[missing_values][column])
        print()
    else:
        print(f"No missing values found in column '{column}'")

# let's plot a histogram for some data

plt.figure(figsize=(10, 6))
for i, column in enumerate(['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']):
    plt.subplot(2, 2, i + 1)
    plt.hist(df[column], bins=20)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
#plt.show()

# plot multi matrix

# Selecting features and target
x = df.iloc[:,:4] # Selecting all rows, first 4 columns
y = df.iloc[:,4] # Selecting all rows, the 5th column

# Combine x and y into a single DataFrame for plotting
data = pd.concat([x, y], axis=1)

# Create a pair plot
sns.pairplot(data, hue='State')
#plt.show()

# let's do a box plot for two selected variables

# Selecting 'R&D Spend' as x and 'Profit' as y
x = df['R&D Spend']
y = df['Profit']

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Define the equation of the line (y = mx + b)
equation = f'y = {slope:.2f}x + {intercept:.2f}'

# Create a scatter plot with regression line
#plt.figure(figsize=(8, 6))
#sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
#plt.title('Scatter Plot with Regression Line (R&D Spend vs Profit)')
#plt.xlabel('R&D Spend')
#plt.ylabel('Profit')

# Add the equation to the plot
#plt.annotate(equation, xy=(x.min(), y.max()), fontsize=12, color='blue')
#plt.show()

# can make predictions with the equation

# let's do a multi linear regression analysis


# selecting independent variables (X1), dependent variable (y1)
X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y = df['Profit']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)

# Calculating the R-squared value
r_squared = r2_score(y_test, y_pred)

# Printing the R-squared value
print(f'R-squared value: {r_squared:.4f}')

# Printing the coefficients and intercept
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# let's do a 3D plot of some selected data

x = df['R&D Spend']
y = df['Marketing Spend']
z = df['Profit']

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o')

# Set labels and title
ax.set_xlabel('R&D Spend')
ax.set_ylabel('Marketing Spend')
ax.set_zlabel('Profit')
plt.title('3D Scatter Plot')

plt.show()

# Selecting features and target
x = df.iloc[:,:4] # Selecting all rows, first 4 columns
y = df.iloc[:,4] # Selecting all rows, the 5th column

# Perform one-hot encoding
ohe = OneHotEncoder(sparse=False) # Output should not be a sparse matrix
x_ohe = ohe.fit_transform(df[['State']]) # Convert data to binary (0, 1) array

# Convert the one-hot encoded array to a DataFrame
x_df = pd.DataFrame(x_ohe, columns=ohe.get_feature_names_out(['State']))

# Apply column transformer
col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'),[0]), # Assuming 'State' is the first column
    remainder='passthrough'
)

x_transformed = col_trans.fit_transform(x_df)

# do a machine learning decision tree prediction

# Selecting features and target
X = df[['R&D Spend', 'Marketing Spend']]  # Adjust features as needed
y = df['Profit']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the decision tree model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = tree_model.predict(X_test)

# Calculating Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(y_test) # actual values
print(y_pred) # predicted values

# lets do k nearest neighbors model

X2 = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y2 = df['Profit']

# Convert categorical variables using one-hot encoding
X2_encoded = pd.get_dummies(X2, columns=['State'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X2_encoded, y2, test_size=0.2, random_state=42)

# Scale the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
knn_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn_model.predict(X_test_scaled)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# Making a single prediction
sample_RnD_Spend = 200000
sample_Administration = 150000
sample_Marketing_Spend = 300000
sample_State = 'California'

# Encode the sample feature values
sample_feature_values = [[sample_RnD_Spend, sample_Administration, sample_Marketing_Spend, 0, 0]]  # Assuming 'California' is the first category
sample_feature_values_scaled = scaler.transform(sample_feature_values)
sample_prediction = knn_model.predict(sample_feature_values_scaled)
print(f'Sample Prediction: {sample_prediction[0]:.2f}') # from the sample features the model predicts the profit for a hypothetical startup

print(x_transformed)
