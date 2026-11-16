from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent
records = [
    [1, 'Yes', 'Single', '125K', 'No'],
    [2, 'No', 'Married', '100K', 'No'],
    [3, 'No', 'Single', '70K', 'No'],
    [4, 'Yes', 'Married', '120K', 'No'],
    [5, 'No', 'Divorced', '95K', 'Yes'],
    [6, 'No', 'Married', '60K', 'No'],
    [7, 'Yes', 'Divorced', '220K', 'No'],
    [8, 'No', 'Single', '85K', 'Yes'],
    [9, 'No', 'Married', '75K', 'No'],
    [10, 'No', 'Single', '90K', 'Yes']
]
df = pd.DataFrame(records, columns=['Tid', 'Refund', 'Marital Status', 'Taxable Income', 'Cheat'])
print('Q1: DataFrame\n', df)
print('Q2: Rows 0,4,7,8\n', df.loc[[0, 4, 7, 8]])
print('Q3a: Rows index 3-7\n', df.iloc[3:8])
print('Q3b: Rows 4-8, columns 2-4\n', df.iloc[4:9, 2:5])
print('Q3c: All rows, columns 1-3\n', df.iloc[:, 1:4])

iris = pd.read_csv(base / 'iris.csv')
print('Q4: Iris first five rows\n', iris.head())
changed = iris.drop(index=4).drop(columns=iris.columns[3])
print('Q5: Drop row 4 and column index 3\n', changed)

employees = pd.read_csv(base / 'employees.csv')
print('Q6a: Shape', employees.shape)
print('Q6b: Information')
employees.info()
print('Q6c: Statistics\n', employees.describe(include='all'))
print('Q6d: Head\n', employees.head(), '\nTail\n', employees.tail(3))
print('Q6e: Average salary', employees['Salary'].mean())
print('Total bonus', employees['Bonus'].sum())
print('Youngest age', employees['Age'].min())
print('Highest rating', employees['Rating'].max())
print('Q6f: Salary descending\n', employees.sort_values('Salary', ascending=False))
employees['Performance'] = employees['Rating'].apply(lambda x: 'Excellent' if x >= 4.5 else 'Good' if x >= 4.0 else 'Average')
print('Q6g: Performance\n', employees[['Name', 'Performance']])
print('Q6h: Missing values\n', employees.isna().sum())
employees = employees.rename(columns={'Employee_ID': 'ID'})
print('Q6i: Renamed columns\n', employees.columns.tolist())
print('Q6j: Experience greater than 5\n', employees[employees['Years_of_Experience'] > 5])
print('IT department\n', employees[employees['Department'] == 'IT'])
employees['Tax'] = employees['Salary'] * 0.10
employees['Salary_After_Tax'] = employees['Salary'] - employees['Tax']
employees.to_csv(base / 'employees_modified.csv', index=False)
print('Q6k-l: Saved employees_modified.csv\n', employees)
