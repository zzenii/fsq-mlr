import pandas as pd # for DataFrame purposes
from tabulate import tabulate # for printable output
from sklearn.linear_model import LinearRegression # to be removed
from sklearn.metrics import mean_squared_error, r2_score # to evaluate model
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib # to import model
from matplotlib import pyplot as plt # for visual output


# Function to translate numerical information (income, age) into ranges to ease analysis.
def assert_ranges(dataframe, column) :
    if column == 'Income' :
        bins = [0, 50000, 75000, 100000, 125000, 150000, 200000]
    elif column == 'Age' :
        bins = [18, 25, 35, 45, 55, 65]
    # Additional columns to set ranges can be added here.

    labels = [f'{bins[x]}-{bins[x+1]}' for x in range(0, len(bins)-1)]

    # Cut function segments the column information into the bins defined above. These bins are attributed labels according to the range of values.
    dataframe[column] = pd.cut(dataframe[column], bins=bins, labels=labels, right=False)
    return dataframe[column]

# Function to calculate percentage of total entries (in a column) represented by a each possible answer. Results are saved in a csv file and outputted using Matplotlib.
def demonstrate_percentages(dataframe, title, csv_label) :
    # 2x3 Subplot Grid
    fig, axs = plt.subplots(2, 3)
    count = 0

    # Iterate by column, use .value_counts to produce pandas series with percentages. Define pie subplot. Save percentages to csv.
    for column in dataframe.columns :
        percentages = dataframe[column].value_counts(normalize=True)*100
        axs[count%2, count//2].pie(percentages, autopct='%1.1f%%')
        axs[count%2, count//2].legend(labels=percentages.index)
        axs[count%2, count//2].set_title(column)
        count+=1
        filename = f'outputs/percentages_{csv_label}_{count}.csv'
        percentages.to_csv(filename, header=column)
    fig.suptitle(title)
    plt.show()
    

#------------------------------------ Part 1: Scikit Linear Regression Model ------------------------------------#

# Read csv data into a Pandas DF
df = pd.read_csv('data/fabricated_survey_data.csv')

# Once survey results are received and placed in a csv, this will look like:
# df = pd.read_csv('data/survey_data.csv)


# Partition into features vs targets & train vs test
X = df[['Income', 'Age', 'Awareness of Charities', 'Confidence in Charities', 'Positive Impact of Charities']]
Y = df[['Donation Amount', 'Donation Frequency']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initiate model
model = LinearRegression()
# Once model is pre-trained, this will look like:
# model = joblib.load('trained_model.pk1')
# Where the model is exported after training using:
# joblib.dump(model, 'trained_model.pk1')

# train and test model
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Mean Squared Error: ", mse)
print("R-squared: ", r2)

# Print results (coefficients of model)
results = pd.DataFrame(model.coef_, columns=['Income', 'Age', 'Awareness of Charities', 'Confidence in Charities', 'Positive Impact of Charities'], index=['Donation Magnitude', 'Donation Frequency'])
results.insert(5, 'Intercepts', model.intercept_)

print(tabulate(results, headers='keys', tablefmt='psql'))

# results to output csv
results.to_csv('outputs/output_coefficients.csv')

#------------------------------------ Part 2: Characteristic Analysis ------------------------------------#

demographic = df[['Income', 'Age', 'Gender', 'Education Level', 'Employment Status', 'Ethnicity', 'Donation Amount', 'Donation Frequency']].sort_values(by='Donation Amount', ascending=False)
demographic['Income'] = assert_ranges(demographic, 'Income')
demographic['Age'] = assert_ranges(demographic, 'Age')

demonstrate_percentages(demographic[['Income', 'Age', 'Gender', 'Education Level', 'Employment Status', 'Ethnicity']].head(25), 'Demographic Characteristics of Top 25 Percent in Donation Amount', 'Top25-Amount')
demonstrate_percentages(demographic[['Income', 'Age', 'Gender', 'Education Level', 'Employment Status', 'Ethnicity']].tail(25), 'Demographic Characteristics of Bottom 25 Percent in Donation Amount', 'Bottom25-Amount')

demographic.sort_values(by='Donation Frequency', ascending=False)
demonstrate_percentages(demographic[['Income', 'Age', 'Gender', 'Education Level', 'Employment Status', 'Ethnicity']].head(25), 'Demographic Characteristics of Top 25 Percent in Donation Frequency', 'Top25-Frequency')
demonstrate_percentages(demographic[['Income', 'Age', 'Gender', 'Education Level', 'Employment Status', 'Ethnicity']].tail(25), 'Demographic Characteristics of Bottom 25 Percent in Donation Frequency', 'Bottom25-Frequency')