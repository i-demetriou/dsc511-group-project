r"""°°°
## Hotel Reviews - Final Project of DSC511
°°°"""
# |%%--%%| <8XxG2ZeFGW|hFGcPSknTF>
r"""°°°
#### Libraries
°°°"""
# |%%--%%| <hFGcPSknTF|ygqwgrGbct>

# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# |%%--%%| <ygqwgrGbct|5MOrmmHwHg>

# Loading dataset

original_data = pd.read_csv('/Users/a35797/Documents/DSC511/Hotel_Reviews.csv')

# |%%--%%| <5MOrmmHwHg|ngIWQCl3Ls>
r"""°°°
#### Understanding the data - EDA
°°°"""
# |%%--%%| <ngIWQCl3Ls|DRrk60fGca>

# Checking with how many rows and columns we have to deal with

original_data.shape

# |%%--%%| <DRrk60fGca|cofkxaaPPJ>
r"""°°°
Our dataset consists of 515,738 observations and 17 features. In order to see the names of the 17 columns (features) we can do the following.
In order to achieve computational efficiency we will take a smaller random sample of the original dataset. This way the code will run faster making it computationally cheaper.
°°°"""
# |%%--%%| <cofkxaaPPJ|G2OtOSELmA>

# Taking as a sample approximately 50k observations

original_sample = original_data.sample(n=50000, random_state=42)  # Change n as needed

# |%%--%%| <G2OtOSELmA|G0GbcDhC7q>

original_sample.columns

# |%%--%%| <G0GbcDhC7q|RGrf49PQzd>

# Checking for duplicates

original_sample.duplicated().any()

# |%%--%%| <RGrf49PQzd|UMwHK8Xsfg>

duplicate_columns = [col for col in original_sample.columns if original_sample[col].duplicated().any()]
print("Columns containing duplicates:", duplicate_columns)

# |%%--%%| <UMwHK8Xsfg|fdW10mnKe8>
r"""°°°
After checking whether the dataset contains duplicates, it turns out that duplicates exist in all of the attributes. To see exactly how many they are we can apply .sum().
°°°"""
# |%%--%%| <fdW10mnKe8|PYzAKosGta>

original_sample.duplicated().sum()

# |%%--%%| <PYzAKosGta|zzgWU3PJ2L>
r"""°°°
The amount of duplicated values corresponds to almost 0% of the whole sample. This is why the duplicates could stay untached without causing any issues in further analysis.
°°°"""
# |%%--%%| <zzgWU3PJ2L|QO3QJf1J2U>

# Seeing the first 5 records of the dataset

original_sample.head()

# |%%--%%| <QO3QJf1J2U|NjRnToaScQ>

# Checking what types of attributes we have

original_sample.dtypes

# |%%--%%| <NjRnToaScQ|b02ngKE7Qt>
r"""°°°
From above it's obvious that some attributes will nedd encoding (i.e Reviewer_Nationality).
°°°"""
# |%%--%%| <b02ngKE7Qt|4D9annLFOg>

# Summary statistics only for numerical features

original_sample.describe()

# |%%--%%| <4D9annLFOg|Mh1p644XL6>
r"""°°°
From the statistical summary above we can decide wheather our data needs scaling or not based on mean and std values. Here mean ranges from 8.4 (΄Average_Score΄) to 496.51 (΄Additional_Number_of_Scoring΄). This is an indication that scaling will be necessary since we are talking about a big enough difference in the scale of our data. Also, it is obvious that numerical feaures do not contain null (NA) values since the count is equal to the number of observations we have in the sample.
°°°"""
# |%%--%%| <Mh1p644XL6|LeNSwb8EXX>

# Checking how many null values each column has
# Mostly necessary for non-numeric features

original_sample.isnull().sum()

# |%%--%%| <LeNSwb8EXX|thLBI7xd1B>
r"""°°°
We can notice that the columns ΄lat΄ and ΄lng΄ hae null values in them. The total number of null values equals to 632 out of 50,000 observations. The 632 observations correspond to 1.26% of the whole sample. This mean that those rows can safely be removed without causing any bias in our results.
°°°"""
# |%%--%%| <thLBI7xd1B|8OUmZWfucR>

# This part is done 'in place'

original_sample.dropna(inplace = True)

# |%%--%%| <8OUmZWfucR|MXjh29fFoi>

# Checking for certainty

original_sample.isnull().any()

# |%%--%%| <MXjh29fFoi|lrJUUBoXjq>
r"""°°°
#### Checking for skewness in the target variable
°°°"""
# |%%--%%| <lrJUUBoXjq|9J8dVeO0N4>

# Creating the histogram

plt.figure(figsize=(8, 4))  # Setting figure size
plt.hist(original_sample['Average_Score'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)

plt.title('Target Variable Distribution', fontsize=14, fontweight='bold', color='#2c3e50')
plt.xlabel('Average Score', fontsize=12, color='#34495e')
plt.ylabel('Frequency', fontsize=12, color='#34495e')

# Remove top and right spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show the plot
plt.show()

# |%%--%%| <9J8dVeO0N4|jP3AXjVOpW>
r"""°°°
From the statistical summary we saw that the mean of the ΄Average_Score΄ is close to 8,4 and from the plot we can see that our target variable is slightly skewed to the left. This could potentially be due to the presence of an outlier around the value 6,3 - 6,4. To be more spesific about the skewness we can use .skew().
°°°"""
# |%%--%%| <jP3AXjVOpW|v3lnrKhnnV>

original_sample['Average_Score'].skew()

# |%%--%%| <v3lnrKhnnV|PmzLQpjrJU>
r"""°°°
#### Checking for linear correlation between numeric features and the target variable
°°°"""
# |%%--%%| <PmzLQpjrJU|2sG7aaZLgf>

!pip install seaborn

# |%%--%%| <2sG7aaZLgf|MMOEmZEgRy>

# Setting Seaborn style for a cleaner look
    
sns.set_style("white")  
sns.set_palette("coolwarm")  

plt.figure(figsize=(12, 8))

# Creating the pairplot
g = sns.pairplot(
    original_sample,
    x_vars=[
        'Additional_Number_of_Scoring', 'Review_Total_Negative_Word_Counts', 
        'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 
        'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score', 'lat', 'lng'
    ],
    y_vars="Average_Score",
    kind="reg",  # Regression line
    plot_kws={'scatter_kws': {'alpha': 0.5, 's': 20, 'color': '#3498db'}},  # Customize scatter points
    height=2.5,  # Adjust individual plot height
)

# Improve spacing
plt.suptitle("Pairplot of numeric features vs. Average Score", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Show the plot
plt.show()

# |%%--%%| <MMOEmZEgRy|Sa7iJ2xFFi>
r"""°°°
It appears that none of the numeric features are linearly correlated to our target variable ΄Average_Score΄ except from ΄Reviewer_Score΄. The points in the 6th plot slope widely around the purple line which is an indication of a weak linear correlation.
°°°"""
# |%%--%%| <Sa7iJ2xFFi|b8BIlUm9ge>
r"""°°°
#### Visualizing correlation using a heatmap
°°°"""
# |%%--%%| <b8BIlUm9ge|kTiau3cNh5>

# An alternative way to visualize linear correlation

fig, ax = plt.subplots()

num_features = original_sample[[
        'Additional_Number_of_Scoring', 'Review_Total_Negative_Word_Counts', 
        'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 
        'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score', 'lat', 'lng', 'Average_Score'
    ]]
sns.heatmap(num_features.corr(method='pearson'), annot=True)

# |%%--%%| <kTiau3cNh5|XWx4kKDFiR>
r"""°°°
From the heatmap we can see that ΄Additional_Number_of_Scoring΄ and ΄Total_Number_of_Reviews΄ are highly linearly correlated since the correlation coefficient is equal to 0.82 (very close to 1). We can also see a weak linera correlation between ΄Additional_Number_of_Scoring΄ and ΄lat΄ where the correlation coefficient is equal to 0,34. As already studied in the previous pairplots ΄Average_Score΄ and ΄Reviewer_Score΄ are linearly correlated even though their correlation is also pretty weak (correlation coefficient equals to 0,37).
°°°"""
# |%%--%%| <XWx4kKDFiR|MIfcDKpiwL>


