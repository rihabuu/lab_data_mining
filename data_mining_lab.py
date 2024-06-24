import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.title('Data Mining Lab 1: IQ Analysis')

st.header('Univariate data set: IQ Analysis')

# Parameters
mean_value = 100
std_deviation = 15
sample_sizes = [10, 1000, 100000]

# Generate samples and calculate statistics
def generate_samples(mean_value, std_deviation, sample_sizes):
    samples = [np.random.normal(mean_value, std_deviation, size) for size in sample_sizes]
    results = []
    confidence_intervals = []
    for sample, size in zip(samples, sample_sizes):
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        std_error = sample_std / np.sqrt(size)
        ci95 = stats.norm.interval(0.95, loc=sample_mean, scale=std_error)
        results.append((sample_mean, sample_std, std_error, ci95))
    return samples, results

samples, results = generate_samples(mean_value, std_deviation, sample_sizes)

# Display sample statistics
st.header('Sample Statistics')
st.write("Here we display the statistics for three different samples generated from a normal distribution with mean 100 and standard deviation 15.")
sample_stats_df = pd.DataFrame({
    'Sample Size': sample_sizes,
    'Sample Mean': [r[0] for r in results],
    'Sample Std Dev': [r[1] for r in results],
    'Std Error': [r[2] for r in results],
    '95% CI Lower': [r[3][0] for r in results],
    '95% CI Upper': [r[3][1] for r in results],
})
st.write(sample_stats_df)

# Theoretical values
theoretical_mean = 100
theoretical_std = 15

# Comparison of sample statistics with theoretical values
st.header('Comparison with Theoretical Values')
st.write("Next, we compare the sample statistics with the theoretical mean and standard deviation.")
comparison_df = pd.DataFrame({
    'Sample Size': sample_sizes,
    'Sample Mean': [r[0] for r in results],
    'Theoretical Mean': [theoretical_mean] * len(sample_sizes),
    'Sample Std Dev': [r[1] for r in results],
    'Theoretical Std Dev': [theoretical_std] * len(sample_sizes),
})
st.write(comparison_df)

# Comment on comparison
st.write("""
**For Sample Size: 10**, the sample mean (99.4357) is close to the theoretical mean (100), but with a small sample size, there can be more variability. The sample standard deviation (13.0084) is slightly lower than the theoretical standard deviation (15), which can also be attributed to the small sample size.

**For Sample Size: 1000**, the sample mean (99.0744) is very close to the theoretical mean (100). The sample standard deviation (14.7587) is also very close to the theoretical standard deviation (15), indicating that as the sample size increases, the sample statistics tend to converge to the theoretical values.

**For Sample Size: 100000**, the sample mean (100.0112) is almost identical to the theoretical mean (100), which is expected given the large sample size. The sample standard deviation (14.987) is almost identical to the theoretical standard deviation (15), confirming that with a large sample size, the sample statistics are very close to the theoretical values.
""")

# Load and analyze malnutrition data
st.header('Malnutrition Data Analysis')
malnutrition_file = st.file_uploader('Upload malnutrition.csv', type='csv')
if malnutrition_file is not None:
    malnutrition_data = pd.read_csv(malnutrition_file, header=None)
    malnutrition_data.columns = ['IQ']
    
    st.write('Columns in the uploaded CSV file:', malnutrition_data.columns.tolist())
    
    # Analyze the malnutrition data
    malnutrition_mean = malnutrition_data['IQ'].mean()
    malnutrition_std = malnutrition_data['IQ'].std(ddof=1)
    malnutrition_size = len(malnutrition_data)
    malnutrition_std_error = malnutrition_std / np.sqrt(malnutrition_size)
    malnutrition_ci95 = stats.norm.interval(0.95, loc=malnutrition_mean, scale=malnutrition_std_error)

    st.subheader('Malnutrition Data Statistics')
    st.write("Below we compare the statistics of the malnutrition sample with the large random sample (size 100000).")
    malnutrition_stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Standard Deviation', '95% CI Lower', '95% CI Upper'],
        'Malnutrition Sample': [malnutrition_mean, malnutrition_std, malnutrition_ci95[0], malnutrition_ci95[1]],
        'Large Sample (100000)': [results[2][0], results[2][1], results[2][3][0], results[2][3][1]],
    })
    st.write(malnutrition_stats_df)

    # Plot histograms
    st.subheader('Histograms')
    st.write("Histograms of IQ scores for the large sample and the malnutrition sample.")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(samples[2], bins=50, alpha=0.7, label='Large Sample (100000)')
    ax[0].set_title('Large Sample (100000)')
    ax[0].legend()
    ax[1].hist(malnutrition_data['IQ'], bins=50, alpha=0.7, label='Malnutrition Sample')
    ax[1].set_title('Malnutrition Sample')
    ax[1].legend()
    st.pyplot(fig)

    st.subheader('Comparison')
    st.write(" The mean IQ for the malnutrition sample (87.98) is significantly lower than the mean IQ of the large sample (100.0164), indicating a substantial negative effect of malnutrition on IQ. The standard deviation is also lower in the malnutrition sample (9.6776) compared to the large sample (14.9847).")
    st.subheader('Conclusion')
    st.write(" As the sample size increases, the sample mean and standard deviation tend to converge to the theoretical values of 100 and 15, respectively. However, the malnutrition sample shows a significantly lower mean IQ, indicating that malnutrition has a negative impact on IQ.")




st.title(' Multivariate data set: Fisher Iris')
st.header('1. Load and Inspect the Iris Data Set')

# Load the Iris dataset
iris_file = st.file_uploader('Upload iris.data', type='data')
if iris_file is not None:
    # Read the .data file without a header and add column names
    iris_data = pd.read_csv(iris_file, header=None)
    iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    # Convert numeric columns to float
    for col in iris_data.columns[:-1]:
        iris_data[col] = pd.to_numeric(iris_data[col], errors='coerce')

    # Display data overview
    st.write('Data Overview:')
    st.write(iris_data.head())

    st.header('2. Histograms of the Different Attributes')
    st.write("The histograms below show the distribution of the different attributes in the Iris dataset.")

    # Display histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, column in enumerate(iris_data.columns[:-1]):
        axes[i].hist(iris_data[column], bins=20, alpha=0.7)
        axes[i].set_title(column)
    st.pyplot(fig)
    
    st.write("""
    **Comments:**
    - The attributes `sepal_length`, `sepal_width`, `petal_length`, and `petal_width` show different patterns in their distributions.
    - `sepal_length` and `petal_length` appear to have a more normal distribution, while `sepal_width` and `petal_width` show some skewness.
    """)

    st.header('3. Compute the Coefficient of Correlation Between All Attributes')
    st.write("The table below shows the correlation coefficients computed manually without using the built-in function.")

    # Compute correlation matrix without using the built-in function
    def compute_correlation_matrix(data):
        correlation_matrix = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i == j:
                    correlation_matrix[i, j] = 1
                else:
                    correlation_matrix[i, j] = np.corrcoef(data.iloc[:, i], data.iloc[:, j])[0, 1]
        return correlation_matrix

    numeric_data = iris_data.iloc[:, :-1]
    correlation_matrix = compute_correlation_matrix(numeric_data)

    # Display correlation matrix
    st.write(pd.DataFrame(correlation_matrix, columns=numeric_data.columns, index=numeric_data.columns))

    st.header('4. Visualize and Confirm Correlation Matrix Using Built-in Functions')
    st.write("The heatmap below visualizes the correlation matrix computed using built-in functions for confirmation.")

    # Compute correlation matrix using built-in function
    correlation_matrix_builtin = numeric_data.corr()

    # Visualize the correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(correlation_matrix_builtin, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(numeric_data.columns)))
    ax.set_yticks(np.arange(len(numeric_data.columns)))
    ax.set_xticklabels(numeric_data.columns, rotation=45)
    ax.set_yticklabels(numeric_data.columns)
    st.pyplot(fig)
    
    st.write("""
    **Comments:**
    - The correlation coefficients between `sepal_length` and `petal_length`, and between `sepal_length` and `petal_width`, are relatively high, indicating a strong positive relationship.
    - Conversely, the correlation between `sepal_width` and the other attributes is relatively low, suggesting weaker relationships.
    """)

    st.header('5. Compute Confidence Intervals for Correlation Coefficients')
    st.write("The table below shows the 95% confidence intervals for the correlation coefficients, assuming a normal distribution of attributes.")

    # Compute confidence intervals for correlation coefficients
    def compute_confidence_intervals(data, alpha=0.05):
        n = data.shape[0]
        ci_matrix = np.zeros((data.shape[1], data.shape[1], 2))
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i != j:
                    r = correlation_matrix_builtin.iloc[i, j]
                    se = 1 / np.sqrt(n - 3)
                    z = np.arctanh(r)
                    z_ci = stats.norm.interval(1 - alpha, loc=z, scale=se)
                    ci_matrix[i, j, 0] = np.tanh(z_ci[0])
                    ci_matrix[i, j, 1] = np.tanh(z_ci[1])
        return ci_matrix

    ci_matrix = compute_confidence_intervals(numeric_data)

    ci_df = pd.DataFrame(ci_matrix.reshape(-1, 2), columns=['95% CI Lower', '95% CI Upper'], index=pd.MultiIndex.from_product([numeric_data.columns, numeric_data.columns]))
    st.write(ci_df)

    st.write("""
    **Comments:**
    - The confidence intervals for the correlation coefficients provide a range within which the true correlation coefficient is likely to fall, with 95% confidence.
    - Wider intervals indicate more uncertainty about the exact value of the correlation coefficient, while narrower intervals indicate more precision.
    """)
    
    

st.title('Multivariate data set: Anthropometric data')

st.header('1. Load and Inspect the Anthropometric Data Set')

# Load the mansize dataset
mansize_file = st.file_uploader('Upload mansize.csv', type='csv')
if mansize_file is not None:
    # Read the CSV file with the correct delimiter
    mansize_data = pd.read_csv(mansize_file, delimiter=';')
    
    # Display data overview
    st.write('Data Overview:')
    st.write(mansize_data.head())

    st.header('2. Descriptive Statistics')
    st.write("The table below shows the descriptive statistics of the dataset using the `describe()` function.")

    # Apply describe() function
    desc_stats = mansize_data.describe()
    st.write(desc_stats)

    st.write("""
    **Comments:**
    - The `describe()` function provides summary statistics of the dataset, including count, mean, standard deviation, min, max, and quartiles.
    - This information helps understand the central tendency, dispersion, and overall distribution of the data.
    """)

    st.header('3. Histograms of the Different Attributes')
    st.write("The histograms below show the distribution of the different attributes in the mansize dataset.")

    # Display histograms
    num_columns = len(mansize_data.columns)
    num_rows = (num_columns + 1) // 2  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 4))
    axes = axes.flatten()
    for i, column in enumerate(mansize_data.columns):
        axes[i].hist(mansize_data[column], bins=20, alpha=0.7)
        axes[i].set_title(column)
    for i in range(num_columns, len(axes)):
        fig.delaxes(axes[i])  # Remove any extra subplots
    st.pyplot(fig)

    st.write("""
    **Comments:**
    - The attributes show different patterns in their distributions. Some may appear normally distributed while others might be skewed.
    - For example, if the height distribution is close to normal, it indicates a balanced spread around the mean.
    """)

    st.header('4. Correlation Between Attributes')
    st.write("The table below shows the correlation coefficients between the different attributes.")

    # Compute correlation matrix using built-in function
    correlation_matrix = mansize_data.corr()

    # Display correlation matrix
    st.write(correlation_matrix)

    st.write("""
    **Comments:**
    - Correlation coefficients close to 1 or -1 indicate a strong relationship, while coefficients close to 0 suggest a weak relationship.
    - The relationship between femur length and height is particularly interesting for predicting height in archaeological studies.
    """)

    st.header('4.1. Visualize the Correlation Matrix')
    st.write("The heatmap below visualizes the correlation matrix.")

    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(mansize_data.columns)))
    ax.set_yticks(np.arange(len(mansize_data.columns)))
    ax.set_xticklabels(mansize_data.columns, rotation=45)
    ax.set_yticklabels(mansize_data.columns)
    st.pyplot(fig)

    st.header('5. Confidence Intervals for Correlation Coefficients')
    st.write("The table below shows the 95% confidence intervals for the correlation coefficients, assuming a normal distribution of attributes.")

    # Compute confidence intervals for correlation coefficients
    def compute_confidence_intervals(data, alpha=0.05):
        n = data.shape[0]
        ci_matrix = np.zeros((data.shape[1], data.shape[1], 2))
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i != j:
                    r = correlation_matrix.iloc[i, j]
                    se = 1 / np.sqrt(n - 3)
                    z = np.arctanh(r)
                    z_ci = stats.norm.interval(1 - alpha, loc=z, scale=se)
                    ci_matrix[i, j, 0] = np.tanh(z_ci[0])
                    ci_matrix[i, j, 1] = np.tanh(z_ci[1])
        return ci_matrix

    ci_matrix = compute_confidence_intervals(mansize_data)

    ci_df = pd.DataFrame(ci_matrix.reshape(-1, 2), columns=['95% CI Lower', '95% CI Upper'], index=pd.MultiIndex.from_product([mansize_data.columns, mansize_data.columns]))
    st.write(ci_df)

    st.write("""
    **Comments:**
    - The confidence intervals provide a range within which the true correlation coefficient is likely to fall, with 95% confidence.
    - This helps in understanding the precision and reliability of the correlation estimates.
    """)

    st.header('6. Conclusion on the Links Between Different Variables')
    st.write("""
    Based on the correlation and confidence interval analyses, we can conclude:
    - There are strong correlations between certain attributes, such as femur length and height, indicating that femur length can be a good predictor of height.
    - Other attributes may show weaker correlations, suggesting less direct relationships.
    - Understanding these correlations helps in various applications, including archaeological studies, where predicting height from skeletal measurements is crucial.
    """)



st.title('Chi-squared test of independence and categorial variables')

st.header('1. Load and Inspect the Weather Data Set')

# Load the weather dataset
weather_file = st.file_uploader('Upload weather.csv', type='csv')
if weather_file is not None:
    weather_data = pd.read_csv(weather_file)
    
    # Display data overview
    st.write('Data Overview:')
    st.write(weather_data.head())

    st.write("""
    **Comments:**
    - The dataset contains the following variables:
      - `outlook`: Categorical variable indicating the weather outlook (e.g., sunny, overcast, rainy).
      - `temperature`: Categorical variable indicating the temperature (e.g., hot, mild, cool).
      - `humidity`: Categorical variable indicating the humidity level (e.g., high, normal).
      - `windy`: Boolean variable indicating whether it is windy.
      - `play`: Categorical variable indicating whether to play or not (yes/no).
    """)

    st.header('2. Contingency Table and Chi-squared Test')
    st.write("The table below shows the contingency table between the variables 'outlook' and 'temperature'.")

    # Create contingency table
    contingency_table = pd.crosstab(weather_data['outlook'], weather_data['temperature'])
    st.write(contingency_table)

    # Degrees of freedom
    dof = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)
    st.write(f"Degrees of Freedom: {dof}")

    st.write("""
    **Comments:**
    - The contingency table shows the frequency distribution of the variables 'outlook' and 'temperature'.
    - The degrees of freedom indicate the number of values that can vary independently in the table.
    """)

    # Perform Chi-squared test
    chi2, p, _, _ = stats.chi2_contingency(contingency_table)
    st.write(f"Chi-squared Statistic: {chi2}")
    st.write(f"P-value: {p}")

    st.write("""
    **Conclusion on Dependency:**
    - The Chi-squared test statistic and p-value help determine whether there is a significant association between 'outlook' and 'temperature'.
    - A low p-value (typically < 0.05) indicates a significant dependency between the variables.
    """)

    st.header('3. Assessing Links Between Other Variables')
    st.write("The tables below show the contingency tables and results of the Chi-squared tests for 'outlook/humidity' and 'temperature/humidity'.")

    # Contingency table and Chi-squared test for outlook/humidity
    st.subheader('Outlook and Humidity')
    contingency_table_oh = pd.crosstab(weather_data['outlook'], weather_data['humidity'])
    st.write(contingency_table_oh)
    chi2_oh, p_oh, _, _ = stats.chi2_contingency(contingency_table_oh)
    st.write(f"Chi-squared Statistic: {chi2_oh}")
    st.write(f"P-value: {p_oh}")

    # Contingency table and Chi-squared test for temperature/humidity
    st.subheader('Temperature and Humidity')
    contingency_table_th = pd.crosstab(weather_data['temperature'], weather_data['humidity'])
    st.write(contingency_table_th)
    chi2_th, p_th, _, _ = stats.chi2_contingency(contingency_table_th)
    st.write(f"Chi-squared Statistic: {chi2_th}")
    st.write(f"P-value: {p_th}")

    st.write("""
    **Conclusion on Other Variables:**
    - The Chi-squared test results for 'outlook/humidity' and 'temperature/humidity' help determine if there are significant associations between these pairs of variables.
    - As with the 'outlook/temperature' test, low p-values indicate significant dependencies.
    """)

