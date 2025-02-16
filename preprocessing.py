import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print("File not found.")
        return None

def count_rows(data):
    return len(data)

def count_columns(data):
    return len(data.columns)

def filter(data, column, upper, lower):
    filtered = data[(data[column] > lower)]
    filtered = filtered[(filtered[column] < upper)]
    return filtered

def mean(data, column):
    return np.mean(data[column])

def std(data, column):
    return np.std(data[column])

def min_value(data, column):
    return np.min(data[column])

def max_value(data, column):
    return np.max(data[column])

def display_summary_table(summary):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=summary.values, 
             colLabels=summary.columns, 
             rowLabels=summary.index, 
             cellLoc='center', 
             loc='center',
             colColours=["#f5f5f5"]*len(summary.columns),
             rowColours=["#f5f5f5"]*len(summary.index))
    
    plt.title('Summary Statistics', fontsize=16)
    plt.show()

def scatter_plot(data, x, y):
    plt.scatter(data[x], data[y], color='blue', alpha=0.7)
    plt.title(f'{x} vs {y}', fontsize=16)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def price_histogram(data, samples):
    bins = int(np.ceil(np.log2(samples) + 1)) 
    print(bins)
    counts, bin_edges, _ = plt.hist(data['price'], bins=bins, color='blue', alpha=0.7)
    plt.xticks(bin_edges.round(1)) 
    plt.title('Price Histogram', fontsize=16)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

def ols_regression(data):
    # Take the log of price and sqrft
    data['log_price'] = np.log(data['price'])
    data['log_sqrft'] = np.log(data['sqrft'])
    
    # Define the explanatory and response variables
    X = data['log_sqrft']  # Explanatory variable
    y = data['log_price']  # Response variable
    
    # Add a constant to the model (for the intercept)
    X = sm.add_constant(X)
    
    # Run the OLS regression
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Print the results summary
    print(results.summary())
    
    # Extract and print key results
    coef = results.params['log_sqrft']
    std_err = results.bse['log_sqrft']
    r_squared = results.rsquared
    
    print("\nCoefficient of log(sqrft):", round(coef, 4))
    print("Standard Error:", round(std_err, 4))
    print("R-squared:", round(r_squared, 4))
    
    # Interpretation
    print("\nInterpretation:")
    print(f"A 1% increase in square footage is associated with a {round(coef, 2)}% change in price.")




Hprice = load_data('Hprice.csv')
# Check if the data was loaded successfully
if Hprice is not None:
    Variables = count_columns(Hprice)
    Samples = count_rows(Hprice)
    Filtered = count_rows(filter(Hprice, 'price', 700, 300))
    print(f"Number of Variables: {Variables}")
    print(f"Number of Samples: {Samples}")
    print(f"Number of Samples after filtering: {Filtered}")
    csv_columns = ['price', 'assess', 'bdrms', 'lotsize', 'sqrft']
    
    #rows = [price, assess, bdrms, lotsize, sqrft]
    #columns = ['Mean', 'SD', 'Min', 'Max', 'Sample Size']
    summary = pd.DataFrame(columns=['Mean', 'SD', 'Min', 'Max', 'Sample Size'], index=csv_columns)
    
    #(row, column) = (elem, 'Mean') in Dataframe
    for elem in csv_columns:
        summary.loc[elem, 'Mean'] = round(mean(Hprice, elem), 2)
        summary.loc[elem, 'SD'] = round(std(Hprice, elem), 2)
        summary.loc[elem, 'Min'] = min_value(Hprice, elem)
        summary.loc[elem, 'Max'] = max_value(Hprice, elem)
        summary.loc[elem, 'Sample Size'] = count_rows(Hprice[elem].dropna())
    

    #print(summary)
    display_summary_table(summary)
    price_histogram(Hprice, Samples)
    scatter_plot(Hprice, 'bdrms', 'price')
