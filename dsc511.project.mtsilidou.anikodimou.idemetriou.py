r"""°°°
# DSC511 Group Project: Hotel Review Sentiment Analysis and Rating Prediction

## Authors

- Maria Tsilidou
- Anastasios Nikodimou
- Ioannis Demetriou
°°°"""
#|%%--%%| <rh9kv9W4Ol|kmBYzlLg4s>
r"""°°°
TODO LIST

- EDA (Exploratory Data Analysis)
- Περισσότερα γραφήματα για κείμενα (π.χ. κατανομές tokens, μέσος όρος λέξεων ανά κείμενο)
- Plots με αριθμό tokens, κατανομή, μέσους όρους κτλ.
- Machine Learning / NLP tasks
- Χρήση περισσότερων αλγορίθμων για ταξινόμηση κειμένου.
- Επεξεργασία κειμένου:
- Lemmatization + Tokenization: εύρεση ποια tokens εμφανίζονται πιο συχνά.
- Classification of Reviews:
- Δοκιμή σε υπάρχοντα reviews αν τα ταξινομεί σωστά ως θετικά ή αρνητικά.
- Δημιουργία δικών μας reviews και δοκιμή ταξινόμησης (π.χ. με χρήση KNN βάσει tokens).
- Bias Analysis
- Έλεγχος για bias στα δεδομένα ή στο μοντέλο (π.χ. imbalance στα labels).
- Scaling / Preprocessing
- Δοκιμή scalers (π.χ. MaxMinScaler) σε NLP χαρακτηριστικά (όπως TF-IDF values).
- Recommendation System
- Ανάπτυξη Hybrid Recommendation System:
- Συνδυασμός content-based και collaborative filtering μεθόδων.
- Να γίνει προσεκτική και σωστή υλοποίηση.
 - Clarify how we plan to interpret the `Additional_Number_of_Scoring` variable.

°°°"""
#|%%--%%| <kmBYzlLg4s|GGUyxopChC>

# Importing libraries

import folium
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import os
import seaborn as sns
import pandas as pd

from nltk.stem import WordNetLemmatizer

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    NaiveBayes,
    MultilayerPerceptronClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    ClusteringEvaluator,
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    NGram,
    RegexTokenizer,
    StopWordsRemover,
    StringIndexer,
    PCA,
    Tokenizer,
    VectorAssembler,
)
from pyspark.ml.recommendation import ALS
from pyspark.ml.regression import LinearRegression

from pyspark.sql import SparkSession
from pyspark.sql.functions import(
    avg,
    array,
    array_contains,
    arrays_zip,
    col,
    concat,
    concat_ws,
    count,
    explode,
    expr,
    isnan,
    isnull,
    length,
    lit,
    lower,
    mean,
    regexp_extract,
    regexp_replace,
    size,
    split,
    sum as Fsum,
    to_date,
    trim,
    udf,
    when,
)
from pyspark.sql.types import ArrayType, BooleanType, FloatType, IntegerType, StringType

sns.set_palette("viridis")
# Get or create a SparkSession object
spark = SparkSession.builder.appName("DSC511-GroupProject").master("local[*]").config("spark.driver.memory", "10g").getOrCreate()

# |%%--%%| <GGUyxopChC|nDMJ7IrrCP>
r"""°°°
## Exploratory Data Analysis

In this section we will load, understand the dimension and schema, and explore our
dataset.

The dataset is obtained from [here](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe).


The csv file contains 17 fields. The description of each field is as below:

- `Hotel_Address`: Address of hotel.
- `Review_Date`: Date when reviewer posted the corresponding review.
- `Average_Score`: Average Score of the hotel, calculated based on the latest comment in the last year.
- `Hotel_Name`: Name of Hotel
- `Reviewer_Nationality`: Nationality of Reviewer
- `Negative_Review`: Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Negative'
- `Review_Total_Negative_Word_Counts`: Total number of words in the negative review.
- `Positive_Review`: Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Positive'
- `Review_Total_Positive_Word_Counts`: Total number of words in the positive review.
- `Reviewer_Score`: Score the reviewer has given to the hotel, based on his/her experience
- `Total_Number_of_Reviews_Reviewer_Has_Given`: Number of Reviews the reviewers has given in the past.
- `Total_Number_of_Reviews`: Total number of valid reviews the hotel has.
- `Tags`: Tags reviewer gave the hotel.
- `days_since_review`: Duration between the review date and scrape date.
- `Additional_Number_of_Scoring`: There are also some guests who just made a scoring on the service rather than a review. This number indicates how many valid scores without review in there.
- `lat`: Latitude of the hotel
- `lng`: longtitude of the hotel

Here we took advantage of spark's ability to understand files compressed with gzip
and we added the dataset in a `.csv.gz` form.
°°°"""
# |%%--%%| <nDMJ7IrrCP|xs7ZaAaLQf>

# Loading dataset

# Note: inferSchema=True is "expensive". Consider removing it for performance if needed
# Consider fitting the once deduced `original_schema` while re-running this

original = spark.read.csv('./data/Hotel_Reviews.csv.gz', header=True, inferSchema=True)
original_schema = original.schema

# |%%--%%| <xs7ZaAaLQf|nX7RFdUSeI>

# Getting an idea about the dataset's shape

original_count = original.count()
print(f'Number of observations: {original_count}')
print(f'Number of features: {len(original_schema)}')

#|%%--%%| <nX7RFdUSeI|0nuKkhm8jM>
r"""°°°
### Understanding the dataset

The dataset appears to be the "exploded" join of a hotel, reviewer and review table. In particular,

Hotel:

- `Hotel_Name`: Name of Hotel
- `Hotel_Address`: Address of hotel.
- `lat`: Latitude of the hotel
- `lng`: longtitude of the hotel
- `Average_Score`: Average Score of the hotel, calculated based on the latest comment in the last year.
- `Total_Number_of_Reviews`: Total number of valid reviews the hotel has.

Reviewer:

- `Reviewer_Nationality`: Nationality of Reviewer
- `Total_Number_of_Reviews_Reviewer_Has_Given`: Number of Reviews the reviewers has given in the past.

Review:

- `Review_Date`: Date when reviewer posted the corresponding review.
- `Negative_Review`: Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Negative'
- `Review_Total_Negative_Word_Counts`: Total number of words in the negative review.
- `Positive_Review`: Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be: 'No Positive'
- `Review_Total_Positive_Word_Counts`: Total number of words in the positive review.
- `Reviewer_Score`: Score the reviewer has given to the hotel, based on his/her experience
- `Tags`: Tags reviewer gave the hotel.
- `days_since_review`: Duration between the review date and scrape date.
- `Additional_Number_of_Scoring`: There are also some guests who just made a scoring on the service rather than a review. This number indicates how many valid scores without review in there.

°°°"""
# |%%--%%| <0nuKkhm8jM|i6c1ifPpWm>

# We visualize the first entries in order to decide our next steps

original.printSchema()

original.select(
    'Hotel_Name', 'Reviewer_Nationality',
    'Review_Date', 'Negative_Review', 'Positive_Review', 'Tags', 'days_since_review'
    ).show(5, truncate=15)

# |%%--%%| <i6c1ifPpWm|8zliPt5I2c>
r"""°°°
### Cleaning the dataset

We want to check the quality of our dataset. Since we are dealing with reviews,
we want to check if:

- There are missing values
- There are duplicate entries
- There are any outliers or noise
- There are obvious erroneous entries
- There are missing features
°°°"""
# |%%--%%| <8zliPt5I2c|b9K4Vmutc1>

cleaned = original

# Checking for missing values

# Counting nulls in each column
missing_counts = cleaned.select([Fsum(col(c).isNull().cast("int")).alias(c) for c in cleaned.columns])

missing_counts.show()

# There are empty values in the dataset but the are not shown here

# |%%--%%| <b9K4Vmutc1|mmWFNXJS3v>
r"""°°°
After further investigating the dataset we noticed that there are empty values that are not treated as null. This may be happening becuase there are empty strings and not null values or they might contain invisible characters (e.g. space). We will replace the empty string with null so that Sparks recognises it and treats it as a missing value.
°°°"""
# |%%--%%| <mmWFNXJS3v|JKtTInWfB6>

# Replacing empty strings with Spark's null
for c in cleaned.columns:
    cleaned = cleaned.withColumn(c, when(trim(col(c)) == '', None).otherwise(col(c)))

# |%%--%%| <JKtTInWfB6|SvhPe2XvbT>

# Rerunning the previous code to check whether the empty strings got indeed converted to null

missing_counts_new = cleaned.select([Fsum(col(c).isNull().cast("int")).alias(c) for c in cleaned.columns])
missing_counts_new.show()

# |%%--%%| <SvhPe2XvbT|bWInBpHXDP>
r"""°°°
Now it is obvious that the columns `Reviewer_Nationality`, `Negative_Review` and `Positive_Review` contain null values that spark can identify.
°°°"""
# |%%--%%| <bWInBpHXDP|xBkdZ6OuZ7>
r"""°°°
Since we trated the empty strings we can deal with null values that have other forms.
In many datasets missing values can be represented as 'NA' (a string) instead of null.
Spark won’t treat 'NA' as a missing value unless we explicitly handle it.
So if we're only checking for null, we might miss those 'fake nulls'.
This is why we need to check each column for the presence of 'fake nulls'.
°°°"""
# |%%--%%| <xBkdZ6OuZ7|Zvi8VwC5o3>

# Define your list of fake/null-like strings
fake_nulls = ['NA', 'N/A', 'na', 'n/a', 'null', 'None', ' N A', ' n a', ' N a']

# Creating an empty list to collect the column names and NA counts
fake_null_counts = []

# Loop through columns and count fake nulls in each
for column in cleaned.columns:
    count = cleaned.select(
        Fsum(when(col(column).isin(fake_nulls), 1).otherwise(0)).alias("fake_null_count")
    ).collect()[0]["fake_null_count"]

    if count > 0:
        fake_null_counts.append((column, count))

# Displaying
spark.createDataFrame(fake_null_counts, ["column", "fake_null_count"]).show(truncate=False)

# |%%--%%| <Zvi8VwC5o3|LcdnYSsle5>
r"""°°°
In total we have 9,917 missing values (1.9% of the whole dataset).
There are more cases of missing values that will be analysed later.
We will also later on convert all these fake null values into null values that Spark
can actually process (just like we did with the empty strings).
°°°"""
#|%%--%%| <LcdnYSsle5|dYiWl3e3kN>

# Checking for duplicates
cleaned = cleaned.drop_duplicates()

print(f'From our dataset, {cleaned.count()} / {original.count()} are distinct')

# |%%--%%| <dYiWl3e3kN|vO0cDou6S3>
r"""°°°
### Type casting the dataset

We now inspect the schema and encoding of our features. We want to cast the data
into a format that will be easier to process.
°°°"""
# |%%--%%| <vO0cDou6S3|PDk76TI1QF>
r"""°°°
From the schema and the first few observations, we see that we can benefit
from casting to more python friendly instances, or categorical features that
can be encoded as such.
°°°"""
#|%%--%%| <PDk76TI1QF|R8IbPQba9m>
r"""°°°
Some datasets use different string literals to mean missing data. These include
'NA', 'No Review', 'N/A', NONE', 'NULL', 'MISSING', '', 0, etc.

Let's harmonize by replacing such data with `None` and cast the coordinates
decimal degrees to float.
°°°"""
# |%%--%%| <R8IbPQba9m|AH7Ufw7iCJ>

# We notice that lon/lat represented here as decimal degrees should be numeric,
# but they are typed as strings. This could mean that there are some "hard-coded" NAs

print('Hotels with missing coordinates')
cleaned\
    .select('Hotel_Address', 'lng', 'lat')\
    .filter(~col('lng').rlike(r'[0-9]') | ~col('lat').rlike(r'[0-9]'))\
    .distinct()\
    .show(truncate=False)

# It seems that 'NA' is `None`. Let's update it
cleaned = cleaned\
    .withColumn(
        'lng',
        when(col('lng') == 'NA', lit(None)).otherwise(col('lng'))
    ).withColumn(
        'lat',
        when(col('lat') == 'NA', lit(None)).otherwise(col('lat'))
    ).withColumn(
        'lng',
        col('lng').cast(FloatType())
    ).withColumn(
        'lat',
        col('lat').cast(FloatType())
    )

# Verify
cleaned\
    .select('Hotel_Address', 'lng', 'lat')\
    .filter(~col('lng').rlike(r'[0-9]') | ~col('lat').rlike(r'[0-9]') | col('lng').isNull() | col('lat').isNull())\
    .distinct()\
    .show(truncate=False)

# Entries with missing lon/lat
print('Number of reviews with missing coordinates:')
cleaned\
    .select('Hotel_Address', 'lng', 'lat')\
    .filter(col('lng').isNull() | col('lat').isNull())\
    .count()

#|%%--%%| <AH7Ufw7iCJ|1O5VQfEk2x>
r"""°°°
The `Hotel_Address` seems to have no missing data. We do this by testing if the
length of the string is less than a reasonable address size and if there is any
`None` values
°°°"""
#|%%--%%| <1O5VQfEk2x|0ckasEA7Vs>
print('Hotel Addresses with None entries:')
cleaned\
    .select('Hotel_Address', 'lng', 'lat')\
    .filter(col('Hotel_Address').isNull() )\
    .count()

print('Hotel Addresses with invalid addresses:')
cleaned\
    .select('Hotel_Address', 'lng', 'lat')\
    .filter(length(col('Hotel_Address')) < 10 )\
    .count()

#|%%--%%| <0ckasEA7Vs|ydraWNJwES>

# According to the dataset, no positive / negative reviews are expressed as 'No Positive' / 'No Negative'
cleaned = cleaned\
    .withColumn(
        'Negative_Review',
        when(col('Negative_Review') == 'No Negative', lit(None)).otherwise(col('Negative_Review'))
    ).withColumn(
        'Positive_Review',
        when(col('Positive_Review') == 'No Positive', lit(None)).otherwise(col('Positive_Review'))
    )

# Verify
cleaned.select('Negative_Review', 'Positive_Review').show()

#|%%--%%| <ydraWNJwES|zebcVMEPiT>
r"""°°°
We notice that some of our features are time related, but are typed as strings.

In particular,

- `Review_Date`
- `days_since_review`

We will cast them to datetime and integers (after we confirm the units) respectively
°°°"""
#|%%--%%| <zebcVMEPiT|R82Hdq7GAZ>

cleaned = cleaned\
    .withColumn('Review_Date', to_date(col('Review_Date'), format='M/d/yyyy'))

# Let's see if it worked
cleaned\
    .select('Review_Date')\
    .show(5)

#|%%--%%| <R82Hdq7GAZ|hgnKyLJ773>

cleaned\
    .select('days_since_review')\
    .show(5)

# We see that the format is "<days> day(s)"

@udf(IntegerType())
def days_ago_udf(literal):
    days, unit = literal.split()
    if not days.isdigit():
        raise RuntimeError(f'Unexpected day: {days}')

    if not unit.startswith('day'):
        raise RuntimeError(f'Unexpected time unit: {unit}')

    try:
        return int(days)
    except Exception as e:
        e.args = (f'An error occurred while processing {literal}: {e}',)
        raise


cleaned = cleaned\
    .withColumn('days_since_review',
        days_ago_udf(col('days_since_review'))
    )

# Admire our result
cleaned\
    .select('days_since_review')\
    .show(5)

#|%%--%%| <hgnKyLJ773|e4JIPhLsTy>

r"""°°°
### Geospatial

In this section we will enrich our data with additional spacial information.
Even though we have have all the coordinates of the hotels, not all hotels
have address, city or country information.

We can use a reverse geocoding server to retrieve the address of each location.
However, because of rate limiting and API keys restrictions we cannot get all
the addresses of all our dataset. Instead, given the democratization of the
spatial data by OpenStreetMap and other affiliated open source champions, we
can host our own Nominatim instance, download the `country_grid.sql.gz` dataset,
and query for features at the specific location.

Please follow the instructions to start the server.

TL;DR: `make nominatim/build && make nominatim/run`
°°°"""
# |%%--%%| <e4JIPhLsTy|QSsRyOLHlW>
import psycopg2

def get_connection():
    try:
        return psycopg2.connect(
            database="nominatim",
            user="postgres",
            password="n7m-geocoding",
            host="localhost",
        )
    except:
        return False

def query_country(lon, lat):
    query = f"""
    SELECT DISTINCT country_osm_grid.country_code
        FROM country_osm_grid
        WHERE ST_Contains(country_osm_grid.geometry, ST_GeomFromText('POINT({lon} {lat})', 4326));
    """

    with nominatim:
        with nominatim.cursor() as curr:
            curr.execute(query)
            return curr.fetchone()[0]


@udf(StringType())
def query_country_udf(lon, lat):
    return query_country(lon, lat)

if os.getenv('DSC511-NOMINATIM'):
    nominatim = get_connection()
    cleaned.withColumn('Country', query_country_udf('lng', 'lat'))

    nominatim.close()
else:
    # Load the processed data to save time
    pass

#|%%--%%| <QSsRyOLHlW|Vx5UQvNCoW>

# We source countries.csv from https://developers.google.com/public-data/docs/canonical/countries_csv
countries = pd.read_csv('./data/countries.csv', delimiter='\t').set_index('ISO')
_countries = countries.name.str.upper().to_dict().items()

@udf(StringType())
def extract_country(address):
    for iso, name in _countries:
        if address.upper().endswith(name):
            return iso

cleaned = cleaned\
    .withColumn('Country', extract_country('Hotel_Address'))


country_indexer = StringIndexer(inputCol='Country', outputCol='Country_Encoded')
cleaned = country_indexer.fit(cleaned).transform(cleaned)
# This is time consuming, so persist
cleaned.persist()

cleaned\
    .select('Country')\
    .groupBy('Country').count().orderBy("count", ascending=False)\
    .show()

# TODO: Do we need cities as well?

#|%%--%%| <Vx5UQvNCoW|R3aUHprpeK>

hotels = cleaned\
    .select('Hotel_Name', 'Hotel_Address', 'lng', 'lat', 'Average_Score', 'Total_Number_of_Reviews')\
    .distinct()

hotels_df = hotels\
    .filter(~col('lng').isNull() & ~col('lat').isNull())\
    .toPandas()

m = folium.Map(location=[33.5, 35.1], zoom_start=4)

def color_review(score):
    if score >=8.0:
        return 'green'
    elif score >= 6.0:
        return 'orange'
    else:
        return 'red'

# TODO: Add additional features

for index, row in hotels_df.iterrows():
    folium.Marker(
        icon=folium.Icon(color=color_review(row['Average_Score'])),
        location=[row['lat'], row['lng']],
        tooltip=row['Hotel_Name'],
        popup=f'<b>{row["Hotel_Name"]}</b><br>Rating: {row["Average_Score"]}',
    ).add_to(m)

m.save('./results/hotels.html')
m

#|%%--%%| <R3aUHprpeK|snsBn8aEaO>
r"""°°°
We notice that the hotels from our dataset come from 6 European cities:

- Vienna, Austria
- Paris, France
- Amsterdam, Netherlands
- Barcelona, Spain
- Milan, Paris
- London, United Kingdom
°°°"""
#|%%--%%| <snsBn8aEaO|hZjMHnVLEW>
r"""°°°
We saw before that our dataset consists of the join between "Hotel", "Reviewer" and "Review".
It is natural to catagorize the "keys" of these tables where possible. In particular,
the natural categorizations are:

- `Reviewer_Nationality`
- `Hotel_Name`

Let's explore and encode them.
°°°"""
#|%%--%%| <hZjMHnVLEW|G2vTRebw98>

byReviewer = cleaned\
    .groupBy('Reviewer_Nationality')

print(f'There are reviewers with {byReviewer.count().count()} different nationalities')
# TODO: According to wikipedia, there are 193 countries. Are there duplicates?

byReviewer\
    .count()\
    .sort('count', ascending=False)\
    .show(n=10, truncate=False)

nationality_indexer = StringIndexer(inputCol='Reviewer_Nationality', outputCol='Reviewer_Nationality_Encoded')
cleaned = nationality_indexer.fit(cleaned).transform(cleaned)
cleaned.select('Reviewer_Nationality', 'Reviewer_Nationality_Encoded').show(5)
#|%%--%%| <G2vTRebw98|w573a28ASO>
r"""°°°
We can see that from our dataset, the British, American and Australian tourists
seem to leave the most reviews. This might be because in our original dataset
all reviews are in English, and in these countries English is their native language.

It's worth noting that this selection was either enforced by the review website
or non-Engish reviews where filtered out before we received it, which appears
to introduce a bias toards more affluent countries.
°°°"""
#|%%--%%| <w573a28ASO|82yXM1VHtf>

byHotel = cleaned\
    .groupBy('Hotel_Name')

print(f'There are reviews from {byHotel.count().count()} different hotels')
# TODO: Check if there are duplicates with similar name

byHotel\
    .count()\
    .sort('count', ascending=False)\
    .show(n=10, truncate=False)

hotel_indexer = StringIndexer(inputCol='Hotel_Name', outputCol='Hotel_Name_Encoded')

cleaned = hotel_indexer.fit(cleaned).transform(cleaned)
cleaned.select('Hotel_Name', 'Hotel_Name_Encoded').show(5)

#|%%--%%| <82yXM1VHtf|tSOGl1DoA4>
r"""°°°
Summary of the cleaned dataset:
- our data needs scaling (mean of `Average_Score` is 8.4 whereas mean of `Total_Number_of_Reviews` is 2744.7)
- text features (reviews) appear to have mean and standard deviation implying that there are numeric characters present.
- The count of `Reviewer_Nationality`, `Negative_Review`, `Positive_Review`, `lat` and `lng` is less than the total number of observations which indicate the presence on missing values in the dataset.
°°°"""
#|%%--%%| <tSOGl1DoA4|A7JTI39CRv>

numerical_features = [
    'Additional_Number_of_Scoring',
    'Average_Score',
    'Review_Total_Negative_Word_Counts',
    'Review_Total_Positive_Word_Counts',
    'Reviewer_Score',
    'Total_Number_of_Reviews',
    'Total_Number_of_Reviews_Reviewer_Has_Given',
]
# At this point preprocessing has finished we save our data to parquet
cleaned.write.parquet('./data/Hotel_Reviews.parquet')
cleaned.summary().show()

# |%%--%%| <A7JTI39CRv|5R3d4gNbbg>
r"""°°°
### Sampling

Our dataset is quite large, and considering all observations when exploring the
dataset would be computational expensive and time consuming.

To overcome this, we sample from the original dataset. Based on our on going analysis,
we aim to generate an as balanced subset as posible
°°°"""
#|%%--%%| <5R3d4gNbbg|LxF4mG1IId>

# Taking a smaller chunk to make exploration more computational efficient

# TODO: Consider taking more targeted sample
# - Try to have equal representation from cities
# - Only consider hotels with a certain number of reviews

sample = cleaned.sample(fraction=0.1, withReplacement=False, seed=42)

# |%%--%%| <LxF4mG1IId|XnhwhjuVqD>
r"""°°°
### Feature Visualization
°°°"""
# |%%--%%| <XnhwhjuVqD|hZiIKJbPyv>

sample_pandas = sample.select(numerical_features).toPandas()

print('[Sample] Numerical features description:')
sample_pandas[numerical_features].describe()

# |%%--%%| <hZiIKJbPyv|izGJVPKzx0>
r"""°°°
#### Checking for skewness in the target variable
°°°"""
# |%%--%%| <izGJVPKzx0|1Wyw8wzBd8>

target_rdd = cleaned.select("Average_Score").rdd\
        .flatMap(lambda x: x)\
        .filter(lambda x: x is not None)

# Computing histogram with 30 bins
bins, counts = target_rdd.histogram(30)
bin_widths = [bins[i+1] - bins[i] for i in range(len(bins)-1)]

# Plot
plt.figure(figsize=(8, 4))
plt.bar(bins[:-1], counts, width=bin_widths, color='#3498db', edgecolor='black', alpha=0.7, align='edge')

plt.title('Target Variable Distribution', fontsize=14, fontweight='bold', color='#2c3e50')
plt.xlabel('Average Score', fontsize=12, color='#34495e')
plt.ylabel('Frequency', fontsize=12, color='#34495e')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

# |%%--%%| <1Wyw8wzBd8|Jj54OdyCV4>

cleaned.select(skewness("Average_Score")).show()

# |%%--%%| <Jj54OdyCV4|bw3NG2DqU4>
r"""°°°
From the statistical summary we saw that the mean of the 'Average_Score' is close to 8.4
and from the plot we can see that our target variable is slightly skewed to the left.

This could potentially be due to the presence of an outlier around the value 6.3 - 6.4.
To be more specific about the skewness we can use .skew().
°°°"""
# |%%--%%| <bw3NG2DqU4|wuMxIcSJmL>

cleaned.select(skewness("Average_Score")).show()

#|%%--%%| <wuMxIcSJmL|7VqL9affQk>
r"""°°°
#### Scatterplots
°°°"""
#|%%--%%| <7VqL9affQk|Rbatj96UNf>

sns.set_style("white")
sns.set_palette("coolwarm")

# Creating the pairplot
g = sns.pairplot(
    sample_pandas,
    x_vars=numerical_features,
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


# |%%--%%| <Rbatj96UNf|O0VfzYyfFv>
r"""°°°
We plotted some numeric features fitted using linear regression against the target variable `Average_Score`.

We notice, as expected, that individual reviewer review `Reviewer_Score`, and the total
"positive" words used in the review `Review_Total_Positive_Word_Counts` is positively correlated
with `Average_Score`. Perphaps more suprising, the more reviews a user has given `Total_Number_of_Reviews_Reviewer_Has_Given`
is also positively correlated with the venue's average score.

On the other side, we again expected that `Review_Total_Negative_Word_Counts` negatively
correlates to a venue's average score. Perphaps more suprising, the more reviews a venue has
`Total_Number_of_Reviews`, the smaller its average score
°°°"""
# |%%--%%| <O0VfzYyfFv|o3sxAs5tEG>
r"""°°°
#### Visualizing correlation using a heatmap
°°°"""
# |%%--%%| <o3sxAs5tEG|vv03SuyONM>

# An alternative way to visualize linear correlation

fig, ax = plt.subplots()

sns.heatmap(sample_pandas[numerical_features].corr(method='pearson'), annot=True)

#|%%--%%| <vv03SuyONM|3B9rANu4lX>
r"""°°°
From the heatmap we can see that `Additional_Number_of_Scoring` and `Total_Number_of_Reviews`
are highly linearly correlated since the correlation coefficient is equal to 0.82 (very close to 1).

We can also see a weak linear correlation between `Additional_Number_of_Scoring`
and ΄lat΄ where the correlation coefficient is equal to 0.34.

As already studied in the previous pairplots `Average_Score` and `Reviewer_Score`
are linearly correlated even though their correlation is also pretty weak
(correlation coefficient equals to 0.37).
°°°"""
# |%%--%%| <3B9rANu4lX|Qh2YO0wwL6>

r"""°°°
#### Density plots
°°°"""
# |%%--%%| <Qh2YO0wwL6|wH1pbt6Wmt>

print(sample_pandas.dtypes)

# |%%--%%| <wH1pbt6Wmt|V209bhLulU>



def print_density_plot(feature):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=sample_pandas, x=col, fill=True, color='#3498db')
    plt.title(f'Density Plot of {feature}', fontsize=14, fontweight='bold')
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


for f in numerical_features:
    print_density_plot(f)

# |%%--%%| <V209bhLulU|rNGeLDAPUY>
r"""°°°
From the density plots we can see that `Additional_Number_of_Scoring` is
skewed to the right potentialy due to the presence of extremely high values (outliers).

The same positive skew and outliers are observed in the `Total_Number_of_Reviews`.
The highest consentration of observations is observed around 12500.

`Review_Total_Positive_Word_Counts`, `Review_Total_Negative_Word_Counts` and
`Total_Number_of_Reviews_Reviewer_Has_Given` are also skewed to the right whereas
`Reviewer_Score` is negatively skewed with most od its data points consentrating around 9.8.
°°°"""
# |%%--%%| <rNGeLDAPUY|ex1QxkWhAl>
r"""°°°
#### Boxplots of features with the target variable

°°°"""
# |%%--%%| <ex1QxkWhAl|0hHe5PjCNV>

def print_boxplot(feature):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="Average_Score", y=col, data=sample_pandas, palette="coolwarm")
    plt.title(f'{feature} by Average_Score', fontsize=14)
    plt.xlabel("Average_Score")
    plt.ylabel(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


print_boxplot('Review_Total_Negative_Word_Counts')
print_boxplot('Review_Total_Positive_Word_Counts')

# |%%--%%| <0hHe5PjCNV|FdYqg7Z7JF>
r"""°°°
`Review_Total_Negative_Word_Counts` by `Average_Score`:
 - There’s a slight downward trend: as `Average_Score` increases, the number of negative words tends to decrease.
 - The median is generally lower for higher scores.
 - The distribution is very wide at each score — especially for mid-range scores (e.g., 7.0 to 8.5), meaning high variability in how many negative words are used regardless of rating.
 - Many outliers, especially in lower scores, suggest some extreme reviews with very high negativity.

`Review_Total_Positive_Word_Counts` by `Average_Score`:
 - Positive trend: as `Average_Score` increases, the number of positive words also increases.
 - The median positive word count rises steadily from low to high scores.
 - Whiskers and outliers increase with higher scores — people who are more satisfied tend to say more and more positive things.

°°°"""
# |%%--%%| <FdYqg7Z7JF|MKpEWwNrx8>
r"""°°°
### Natural Language Processing
°°°"""
# |%%--%%| <MKpEWwNrx8|iDYGqhvTPU>

cleaned.select("Negative_Review", "Positive_Review").show(3, truncate=False)

# |%%--%%| <iDYGqhvTPU|fqTGZYVu8T>

# We create two new columns: 'Negative_Word_Count' and 'Positive_Word_Count'
# If a review is null, we set the word count to 0
# Otherwise, we trim the review text, split it into words, and count the number of words

cleaned = cleaned.withColumn(
    "Negative_Word_Count",
    when(col("Negative_Review").isNull(), 0)
    .otherwise(size(split(trim(col("Negative_Review")), "\\s+")))
).withColumn(
    "Positive_Word_Count",
    when(col("Positive_Review").isNull(), 0)
    .otherwise(size(split(trim(col("Positive_Review")), "\\s+")))
)

# |%%--%%| <fqTGZYVu8T|03dfxMyL5G>

cleaned.select(
    "Negative_Review",
    "Negative_Word_Count",
    "Positive_Review",
    "Positive_Word_Count",
    "Review_Total_Negative_Word_Counts",
    "Review_Total_Positive_Word_Counts"
).show(5, truncate=False)

# We select and display the review texts, the word counts we calculated, and the original total word count columns
# This allows us to visually compare our new word counts with the existing 'Review_Total_Negative_Word_Counts' and 'Review_Total_Positive_Word_Counts'

# |%%--%%| <03dfxMyL5G|HlAvfmDJjl>
r"""°°°
We created two new columns that count the actual number of words in each review.
The existing columns from the original dataset include extra blanks, so their word
counts may be inaccurate.
°°°"""
# |%%--%%| <HlAvfmDJjl|6MWKBgMnDf>
r"""°°°
### Text pre-processing

Before applying any advanced analytics methods to our data, in order to execute text analytics,
we need to pre-process our text. Usually this step is consisted by the methods we shall see below

°°°"""
# |%%--%%| <6MWKBgMnDf|Z2lDYjgEmA>
r"""°°°
#### 1.5.1 Tokenization
°°°"""
# |%%--%%| <Z2lDYjgEmA|ixYa14E7Jc>

#Step 1:
cleaned = cleaned.withColumn(
    "Negative_Review_trimmed", when(col("Negative_Review").isNull(), None)
                               .otherwise(trim(col("Negative_Review")))
).withColumn(
    "Positive_Review_trimmed", when(col("Positive_Review").isNull(), None)
                               .otherwise(trim(col("Positive_Review")))
)

# Step 2: Replace nulls with "" for RegexTokenizer
cleaned = cleaned.withColumn(
    "Negative_Review_safe", when(col("Negative_Review_trimmed").isNull(), "").otherwise(col("Negative_Review_trimmed"))
).withColumn(
    "Positive_Review_safe", when(col("Positive_Review_trimmed").isNull(), "").otherwise(col("Positive_Review_trimmed"))
)

# Step 3: Apply RegexTokenizer with pattern \W (non-word characters)
regex_tokenizer_neg = RegexTokenizer(inputCol="Negative_Review_safe", outputCol="Negative_Tokens_Array", pattern="\\W")
regex_tokenizer_pos = RegexTokenizer(inputCol="Positive_Review_safe", outputCol="Positive_Tokens_Array", pattern="\\W")

cleaned = regex_tokenizer_neg.transform(cleaned)
cleaned = regex_tokenizer_pos.transform(cleaned)

# Step 4: Optional – Count tokens
countTokens = udf(lambda tokens: len(tokens) if tokens else None, IntegerType())

cleaned = cleaned.withColumn("Negative_Tokens_Count", when(col("Negative_Review").isNull(), None).otherwise(countTokens(col("Negative_Tokens_Array")))) \
                 .withColumn("Positive_Tokens_Count", when(col("Positive_Review").isNull(), None).otherwise(countTokens(col("Positive_Tokens_Array"))))

# Step 5: Show results
cleaned.select("Negative_Review", "Negative_Tokens_Array", "Negative_Tokens_Count",
               "Positive_Review", "Positive_Tokens_Array", "Positive_Tokens_Count").show(5, truncate=False)

# Prepare reviews: trim, replace nulls, split into words, countwords, and display
# And for the column 'Tags' maybe we should use the regex tokenizer

# |%%--%%| <ixYa14E7Jc|hzF72RvG36>
r"""°°°
#### 1.5.2 Text Cleanup
°°°"""
# |%%--%%| <hzF72RvG36|rOhTFLpsx6>

def clean_text(c):
  c = lower(c)
  c = regexp_replace(c, ",", " ") # Replace commas with spaces
  c = regexp_replace(c, "[^a-zA-Z0-9\\s]", "") # Keep only alphabetic characters and numbers
  return c

# sentenceDataFrame is the dataframe we defined in our last running cell
# Apply clean_text() to multiple columns
clean_text_df = cleaned.select(
    clean_text(col("Positive_Review")).alias("Positive_Review"),
    clean_text(col("Negative_Review")).alias("Negative_Review"),
    clean_text(col("Hotel_Address")).alias("Hotel_Address"),
    clean_text(col("Hotel_Name")).alias("Hotel_Name"),
    clean_text(col("Tags")).alias("Tags"),
    clean_text(col("Reviewer_Nationality")).alias("Reviewer_Nationality")
)

clean_text_df.printSchema()
clean_text_df.show(20)

# |%%--%%| <rOhTFLpsx6|6IMSTqeyop>
r"""°°°
We observe that some reviews contain "no" and "negative" as tokens, but they are
actually real negative reviews. For example:
> "The only negative with the room was there was no mirror outside of the bathroom."

On the other hand, some reviews that contain "no negative" actually mean that there
are no negative comments. For example:
> "I have no negative comments."

Based on this observation, we are going to clean the dataset by setting to null
to the negative reviews where "no" and "negative" appear consecutively as tokens.
°°°"""
# |%%--%%| <6IMSTqeyop|J4zXoV1ALW>

# Define a UDF to check if the sequence "not at all" appears in the tokens
@udf(BooleanType())
def has_not_at_all_udf(tokens):
    if tokens is None or len(tokens) < 3:
        return False
    return any(tokens[i] == "not" and tokens[i+1] == "at" and tokens[i+2] == "all" for i in range(len(tokens) - 2))

# Now use it to filter
cleaned.filter(has_not_at_all_udf(col("Negative_Tokens_Array"))) \
       .select("Hotel_Name", "Negative_Review", "Negative_Tokens_Array") \
       .show(20, truncate=False)

# |%%--%%| <J4zXoV1ALW|NTwnY36NIA>

# Count reviews where tokens start with ["not", "at", "all"] and length is 3 or 4
count_not_at_all = cleaned.filter(
    (size(col("Negative_Tokens_Array")).isin(3, 4)) &  # Exactly 3 or 4 5 tokens
    (col("Negative_Tokens_Array")[0] == "not") &
    (col("Negative_Tokens_Array")[1] == "at") &
    (col("Negative_Tokens_Array")[2] == "all")
).count()

print(f"Number of reviews starting with tokens ['not', 'at', 'all'] and size 3 or 4: {count_not_at_all}")

# |%%--%%| <NTwnY36NIA|RYoaFWOdvK>

# Replace reviews that start with "not at all" and have exactly 3 or 4 tokens
cleaned = cleaned.withColumn(
    "Negative_Review",
    when(
        (size(col("Negative_Tokens_Array")).isin(3, 4)) &
        (col("Negative_Tokens_Array")[0] == "not") &
        (col("Negative_Tokens_Array")[1] == "at") &
        (col("Negative_Tokens_Array")[2] == "all"),
        None  # Set review to null
    ).otherwise(col("Negative_Review"))
).withColumn(
    "Negative_Tokens_Array",
    when(
        (size(col("Negative_Tokens_Array")).isin(3, 4)) &
        (col("Negative_Tokens_Array")[0] == "not") &
        (col("Negative_Tokens_Array")[1] == "at") &
        (col("Negative_Tokens_Array")[2] == "all"),
        array()  # Set tokens to empty array []
    ).otherwise(col("Negative_Tokens_Array"))
).withColumn(
    "Negative_Tokens_Count",
    when(
        (size(col("Negative_Tokens_Array")).isin(3, 4)) &
        (col("Negative_Tokens_Array")[0] == "not") &
        (col("Negative_Tokens_Array")[1] == "at") &
        (col("Negative_Tokens_Array")[2] == "all"),
        lit(0)  # Set token count to 0
    ).otherwise(col("Negative_Tokens_Count"))
)

# |%%--%%| <RYoaFWOdvK|khPFjtcoY7>

# Rerunning the code to verify that the reviews mentioned above have now been replaced
count_not_at_all = cleaned.filter(
    (size(col("Negative_Tokens_Array")).isin(3, 4)) &  # Exactly 3 or 4 5 tokens
    (col("Negative_Tokens_Array")[0] == "not") &
    (col("Negative_Tokens_Array")[1] == "at") &
    (col("Negative_Tokens_Array")[2] == "all")
).count()

print(f"Number of reviews starting with tokens ['not', 'at', 'all'] and size 3 or 4: {count_not_at_all}")

# |%%--%%| <khPFjtcoY7|YdiyTkERDa>
r"""°°°
Here, we essentially want to consider only reviews where the negative review starts with "not at all". However, we restrict the number of tokens to exactly 3 or 4, in order to include only short reviews that consist solely of these tokens.

For example, reviews like "Club staff was not at all friendly at all in fact" should not be counted, because they have more than 4 tokens and "not at all" is not the entire review — and these are actually negative reviews.
°°°"""
# |%%--%%| <YdiyTkERDa|KS5u2BrNyk>

# Find and count reviews where:
# - Negative_Tokens_Array contains "nothing"
# - and the number of tokens is exactly 1 or 2
count_nothing_restricted = cleaned.filter(
    array_contains(col("Negative_Tokens_Array"), "nothing") &
    (size(col("Negative_Tokens_Array")).isin(1, 2))
).count()

print(f"Number of reviews containing the token 'nothing' with 1 or 2 tokens: {count_nothing_restricted}")

# |%%--%%| <KS5u2BrNyk|RxU5ulpI6K>
r"""°°°
These reviews do not actually contain anything negative and should be considered as null. Their tokens should also be set to an empty array (zero tokens), so that the models we build later do not make decisions based on them.
°°°"""
# |%%--%%| <RxU5ulpI6K|d2YWszcrYv>

# Replace reviews containing only "nothing" (with 1 or 2 tokens) with null reviews and empty token arrays
cleaned = cleaned.withColumn(
    "Negative_Review",
    when(
        array_contains(col("Negative_Tokens_Array"), "nothing") & size(col("Negative_Tokens_Array")).isin(1, 2),
        None  # Set review to null
    ).otherwise(col("Negative_Review"))
).withColumn(
    "Negative_Tokens_Array",
    when(
        array_contains(col("Negative_Tokens_Array"), "nothing") & size(col("Negative_Tokens_Array")).isin(1, 2),
        array()  # Set tokens to empty array
    ).otherwise(col("Negative_Tokens_Array"))
).withColumn(
    "Negative_Tokens_Count",
    when(
        array_contains(col("Negative_Tokens_Array"), "nothing") & size(col("Negative_Tokens_Array")).isin(1, 2),
        lit(0)  # Set token count to 0
    ).otherwise(col("Negative_Tokens_Count"))
)

# |%%--%%| <d2YWszcrYv|hP6thnueHV>

# Rerunning the code to verify that the reviews mentioned above have now been replaced

count_nothing_restricted = cleaned.filter(
    array_contains(col("Negative_Tokens_Array"), "nothing") &
    (size(col("Negative_Tokens_Array")).isin(1, 2))
).count()

print(f"Number of reviews containing the token 'nothing' with 1 or 2 tokens: {count_nothing_restricted}")

# |%%--%%| <hP6thnueHV|CKQMd5dXbY>

# Define UDF to check if "no" is followed by "negative"
@udf(BooleanType())
def has_no_negative_pair_udf(tokens):
    if tokens is None or len(tokens) < 2:
        return False
    return any(tokens[i] == "no" and tokens[i+1] == "negative" for i in range(len(tokens) - 1))

# Apply the filter and show rows
cleaned.filter(has_no_negative_pair_udf(col("Negative_Tokens_Array"))) \
    .select("Negative_Review", "Negative_Tokens_Array") \
    .show(20, truncate=False)

# |%%--%%| <CKQMd5dXbY|3ntlodV1mD>

no_negative_count = cleaned.filter(has_no_negative_pair_udf(col("Negative_Tokens_Array"))).count()

print(f"Number of reviews with 'no' followed by 'negative': {no_negative_count}")

# |%%--%%| <3ntlodV1mD|J6EULcLanI>
r"""°°°
Some hotels have non-negative reviews recorded as negative reviews, as their texts contain the consecutive words "no" and "negative." These reviews were not treated as nulls, leading to incorrect labeling and introducing bias into the dataset.
°°°"""
# |%%--%%| <J6EULcLanI|XovnntZw4y>

# Set Negative_Review to null, Negative_Tokens_Array to empty array, and Negative_Tokens_Count to 0 if "no negative" appears
cleaned = cleaned.withColumn(
    "Negative_Review",
    when(
        has_no_negative_pair_udf(col("Negative_Tokens_Array")),
        None  # Set review to null
    ).otherwise(col("Negative_Review"))
).withColumn(
    "Negative_Tokens_Array",
    when(
        has_no_negative_pair_udf(col("Negative_Tokens_Array")),
        array()  # Set tokens to empty array []
    ).otherwise(col("Negative_Tokens_Array"))
).withColumn(
    "Negative_Tokens_Count",
    when(
        has_no_negative_pair_udf(col("Negative_Tokens_Array")),
        lit(0)  # Set token count to 0
    ).otherwise(col("Negative_Tokens_Count"))
)

# |%%--%%| <XovnntZw4y|2ceV5A8zVm>

# Rerunning the code to check if the records we observed have been replaced

no_negative_count = cleaned.filter(has_no_negative_pair_udf(col("Negative_Tokens_Array"))).count()

print(f"Number of reviews with 'no' followed by 'negative': {no_negative_count}")

# |%%--%%| <2ceV5A8zVm|sQJVLdOluf>
r"""°°°
Now we are going to clean the positive reviews ( same process as above)
°°°"""
# |%%--%%| <sQJVLdOluf|d2rJfIdJgD>

# Define a UDF to check if "no" is immediately followed by "positive" in Positive_Tokens_Array
@udf(BooleanType())
def has_no_positive_pair_udf(tokens):
    if tokens is None or len(tokens) < 2:
        return False
    return any(tokens[i] == "no" and tokens[i+1] == "positive" for i in range(len(tokens) - 1))

# Apply the filter
cleaned.filter(
    has_no_positive_pair_udf(col("Positive_Tokens_Array"))
).select("Positive_Review", "Positive_Tokens_Array") \
 .show(20, truncate=False)

# We filter the dataset to find reviews where "no" is immediately followed by "positive" in the Positive_Tokens_Array
# Then we select and display the original positive review text and its tokens for inspection

# |%%--%%| <d2rJfIdJgD|OPD7O8MiEo>

# Count positive reviews where tokens start with ["not", "at", "all"] and length is 3 or 4
count_not_at_all_positive = cleaned.filter(
    (size(col("Positive_Tokens_Array")).isin(3, 4)) &  # Exactly 3 or 4 tokens
    (col("Positive_Tokens_Array")[0] == "not") &
    (col("Positive_Tokens_Array")[1] == "at") &
    (col("Positive_Tokens_Array")[2] == "all")
).count()

print(f"Number of positive reviews starting with tokens ['not', 'at', 'all'] and size 3 or 4: {count_not_at_all_positive}")

# We count the number of positive reviews that start with "not at all" and have exactly 3 or 4 tokens

# |%%--%%| <OPD7O8MiEo|zj4zSLz9sa>

# Find and count positive reviews where:
# - Positive_Tokens_Array contains "nothing"
# - and the number of tokens is exactly 1 or 2
count_nothing_restricted_positive = cleaned.filter(
    array_contains(col("Positive_Tokens_Array"), "nothing") &
    (size(col("Positive_Tokens_Array")).isin(1, 2))
).count()

print(f"Number of positive reviews containing the token 'nothing' with 1 or 2 tokens: {count_nothing_restricted_positive}")

# We count positive reviews that only say "nothing" and have exactly 1 or 2 tokens

# |%%--%%| <zj4zSLz9sa|tX78VWAVkK>
r"""°°°
The number of positive reviews that do not correspond to actual positive reviews is smaller compared to the negative reviews we saw earlier. Therefore, we will only replace with missing values the observations that have no comment, and handle only those specific cases.(above code)
°°°"""
# |%%--%%| <tX78VWAVkK|eSfwgHXNJS>

# Replace positive reviews where:
# - Positive_Tokens_Array contains "nothing"
# - and the number of tokens is exactly 1 or 2
cleaned = cleaned.withColumn(
    "Positive_Review",
    when(
        array_contains(col("Positive_Tokens_Array"), "nothing") & size(col("Positive_Tokens_Array")).isin(1, 2),
        None  # Set review to null
    ).otherwise(col("Positive_Review"))
).withColumn(
    "Positive_Tokens_Array",
    when(
        array_contains(col("Positive_Tokens_Array"), "nothing") & size(col("Positive_Tokens_Array")).isin(1, 2),
        array()  # Set tokens to empty array []
    ).otherwise(col("Positive_Tokens_Array"))
).withColumn(
    "Positive_Tokens_Count",
    when(
        array_contains(col("Positive_Tokens_Array"), "nothing") & size(col("Positive_Tokens_Array")).isin(1, 2),
        lit(0)  # Set token count to 0
    ).otherwise(col("Positive_Tokens_Count"))
)

# We replace positive reviews containing only "nothing" with null reviews and empty token arrays

# |%%--%%| <eSfwgHXNJS|BbyQO62Sky>

#Rerruning the code
count_nothing_restricted_positive = cleaned.filter(
    array_contains(col("Positive_Tokens_Array"), "nothing") &
    (size(col("Positive_Tokens_Array")).isin(1, 2))
).count()

print(f"Number of positive reviews containing the token 'nothing' with 1 or 2 tokens: {count_nothing_restricted_positive}")

# We count positive reviews that only say "nothing" and have exactly 1 or 2 tokens

# |%%--%%| <BbyQO62Sky|CA82Y2Sm22>
r"""°°°
It is important to note that the previous step of cleaning some reviews,
which were considered neither negative nor positive and contained the word "no,"
must take place at this part of the code.

This is because, after removing the stop words, "no" would have been removed.
°°°"""
# |%%--%%| <CA82Y2Sm22|l2kfHyruOU>
r"""°°°
### Removing Stopwords
°°°"""
# |%%--%%| <l2kfHyruOU|Cvznptvjmw>
r"""°°°
Stopwords are words such as "and" and "the" that typically do not add value to the
semantic meaning of a sentence. We typically want to remove these as a means to
reduce the noise in our text datasets.
°°°"""
# |%%--%%| <Cvznptvjmw|EsXzqaSOMW>

# Step 4: Remove Stopwords
stopwords_remover_neg = StopWordsRemover(inputCol="Negative_Tokens_Array", outputCol="Negative_Filtered_Tokens")
stopwords_remover_pos = StopWordsRemover(inputCol="Positive_Tokens_Array", outputCol="Positive_Filtered_Tokens")

cleaned = stopwords_remover_neg.transform(cleaned)
cleaned = stopwords_remover_pos.transform(cleaned)

# Step 5: Count tokens (after stopword removal)
countTokens = udf(lambda tokens: len(tokens) if tokens else None, IntegerType())

cleaned = cleaned.withColumn(
    "Negative_Tokens_Count", when(col("Negative_Review").isNull(), None).otherwise(countTokens(col("Negative_Filtered_Tokens")))
).withColumn(
    "Positive_Tokens_Count", when(col("Positive_Review").isNull(), None).otherwise(countTokens(col("Positive_Filtered_Tokens")))
)

# Step 6: Show results
cleaned.select(
    "Negative_Review", "Negative_Filtered_Tokens", "Negative_Tokens_Count",
    "Positive_Review", "Positive_Filtered_Tokens", "Positive_Tokens_Count"
).show(5, truncate=False)

# |%%--%%| <EsXzqaSOMW|cnsJdV7Nfm>
r"""°°°
### 1.7 Lemmatizing
°°°"""
# |%%--%%| <cnsJdV7Nfm|BeIKNXPHau>

#Install NLTK and download required resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# |%%--%%| <BeIKNXPHau|mZSAj6G6RC>

#Import and define the lemmatizer + UDF:

lemmatizer = WordNetLemmatizer()

# Python function for lemmatizing
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens] if tokens else []

# PySpark UDF
lemmer_udf = udf(lemmatize_tokens, ArrayType(StringType()))

# |%%--%%| <mZSAj6G6RC|5my9CO1kFM>

#Apply lemmatization to the filtered token columns:

cleaned = cleaned.withColumn("Negative_Lemmatized", lemmer_udf(col("Negative_Filtered_Tokens")))
cleaned = cleaned.withColumn("Positive_Lemmatized", lemmer_udf(col("Positive_Filtered_Tokens")))

# |%%--%%| <5my9CO1kFM|FW8V7mvDcy>

#Re-count after lemmatization:
cleaned = cleaned.withColumn("Negative_Lemma_Count", countTokens(col("Negative_Lemmatized")))
cleaned = cleaned.withColumn("Positive_Lemma_Count", countTokens(col("Positive_Lemmatized")))

# |%%--%%| <FW8V7mvDcy|k1SvN61XfL>

#final columns
cleaned.select(
    "Negative_Review", "Negative_Filtered_Tokens", "Negative_Lemmatized", "Negative_Lemma_Count",
    "Positive_Review", "Positive_Filtered_Tokens", "Positive_Lemmatized", "Positive_Lemma_Count"
).show(5, truncate=False)

# |%%--%%| <k1SvN61XfL|LgECY1WVCo>

# Count how many rows have nulls in Positive_Tokens_Count
positive_nulls = cleaned.filter(col("Positive_Tokens_Count").isNull()).count()

# Count how many rows have nulls in Negative_Tokens_Count
negative_nulls = cleaned.filter(col("Negative_Tokens_Count").isNull()).count()

print(f"Number of nulls in Positive_Tokens_Count: {positive_nulls}")
print(f"Number of nulls in Negative_Tokens_Count: {negative_nulls}")

# |%%--%%| <LgECY1WVCo|69xZ8As5D5>

# Count nulls in Negative_Tokens_Count
nulls_negative_tokens = cleaned.filter(col("Negative_Lemma_Count").isNull()).count()

# Count nulls in Positive_Tokens_Count
nulls_positive_tokens = cleaned.filter(col("Positive_Lemma_Count").isNull()).count()

print(f"Number of nulls in Negative_Tokens_Count: {nulls_negative_tokens}")
print(f"Number of nulls in Positive_Tokens_Count: {nulls_positive_tokens}")

# |%%--%%| <69xZ8As5D5|PzmfJfvq20>
r"""°°°
#### Graph Analytic
°°°"""
# |%%--%%| <PzmfJfvq20|iQKrhFjeZw>

df = cleaned.withColumn(
    "reviewer_id",
    concat_ws("_",
        col("Reviewer_Nationality"),
        col("Tags"),
        col("Total_Number_of_Reviews_Reviewer_Has_Given").cast("string")
    )
)

# Build edge list: reviewer → hotel
edge_list = df.select(
    df["reviewer_id"].alias("src"),
    df["Hotel_Name"].alias("dst"),
    df["Reviewer_Score"].alias("weight")
)

# Graph visualization

def plot_directed_graph(edge_list, weighted=True):
    plt.figure(figsize=(12, 9))
    gPlot = nx.DiGraph()
    edge_labels = {}

    edge_char = "weight" if weighted else "relationship"

    # Add edges (limit for visualization)
    for row in edge_list.select("src", "dst", edge_char).take(1000):
        gPlot.add_edge(row["src"], row["dst"])
        edge_labels[(row["src"], row["dst"])] = row[edge_char]

    # Layout and drawing
    pos = nx.spring_layout(gPlot, k=0.15, iterations=20)
    nx.draw(gPlot, pos, with_labels=False, node_size=50, edge_color='gray')
    nx.draw_networkx_edge_labels(gPlot, pos, edge_labels=edge_labels, font_color="green", font_size=8)

    plt.title("Reviewer–Hotel Bipartite Graph (Profile-Based Reviewer ID)")
    plt.show()

# |%%--%%| <iQKrhFjeZw|AILa7ygb6S>

plot_directed_graph(edge_list, weighted=True)

# |%%--%%| <AILa7ygb6S|lV6H9mn65K>
r"""°°°
Bipartite Graph: Reviewers ↔ Hotels
Nodes:
- One set = Reviewers (can be uniquely identified by reviewer + nationality, or anonymized ID)
- Other set = Hotels
- Edges: A review (with edge attributes like score, date, sentiment, tags, etc.)\

Use cases:
- Collaborative filtering for recommendations (like "reviewers similar to you liked...")
- Identify highly connected hotels (most reviewed or diverse set of reviewers)
- Detect reviewer communities or suspicious review patterns (e.g., cliques)
°°°"""
# |%%--%%| <lV6H9mn65K|uxL1SD0i7p>

gPlot = nx.DiGraph()

for row in edge_list.select("src", "dst", "weight").collect():
    gPlot.add_edge(row["src"], row["dst"], weight=row["weight"])

# |%%--%%| <uxL1SD0i7p|fNYmeHGlGS>
r"""°°°
The following code basically ranks a list of popular hotels by how many people reviewed them.
°°°"""
# |%%--%%| <fNYmeHGlGS|r8V2cpEEd6>

hotel_degrees = [(n, gPlot.degree(n)) for n in gPlot.nodes if "Hotel" in n]
top_hotels = sorted(hotel_degrees, key=lambda x: x[1], reverse=True)[:10]
print("Top-reviewed hotels:", top_hotels)

# |%%--%%| <r8V2cpEEd6|3HIKdO3OXV>

# This need too much time to run so maybe we can just remove it

#from networkx.algorithms import community
#communities = community.greedy_modularity_communities(gPlot.to_undirected())
#print("Found", len(communities), "communities")

# |%%--%%| <3HIKdO3OXV|kZyyuXaIln>

top_reviewers = sorted(
    [(n, gPlot.degree(n)) for n in gPlot.nodes if "Hotel" not in n],
    key=lambda x: x[1], reverse=True
)[:10]
print("Most active reviewer profiles:", top_reviewers)

# |%%--%%| <kZyyuXaIln|ZHo73FF0I0>
r"""°°°
#### Recommendation systems - Collaborative filtering
°°°"""
# |%%--%%| <ZHo73FF0I0|bobv25JcoG>

# Step 1: Create a numeric version of reviewer_id and Hotel_Name
reviewer_indexer = StringIndexer(inputCol="reviewer_id", outputCol="reviewer_id_index")
hotel_indexer = StringIndexer(inputCol="Hotel_Name", outputCol="hotel_id_index")

# Fit and transform both
df = reviewer_indexer.fit(df).transform(df)
df = hotel_indexer.fit(df).transform(df)

# Step 2: Split the dataset
(training_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

# Step 3: Build and train ALS model
als = ALS(
    maxIter=10,
    regParam=0.01,
    userCol="reviewer_id_index",
    itemCol="hotel_id_index",
    ratingCol="Reviewer_Score",
    coldStartStrategy="drop"
)

model = als.fit(training_data)

# Step 4: Generate top 10 hotel recommendations for each reviewer
user_recs = model.recommendForAllUsers(10)

# Step 5: Flatten nested recommendations into rows
user_recs = user_recs.selectExpr("reviewer_id_index", "explode(recommendations) as recommendation")
user_recs = user_recs.selectExpr(
    "reviewer_id_index",
    "recommendation.hotel_id_index as hotel_id_index",
    "recommendation.rating as rating"
)

user_recs.show(truncate=False)

# |%%--%%| <bobv25JcoG|WQiAUAaTAM>
r"""°°°
#### RMSE Evaluation
°°°"""
# |%%--%%| <WQiAUAaTAM|ykMJT7653q>

predictions = model.transform(test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Reviewer_Score", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
#|%%--%%| <ykMJT7653q|zF6H9fHwAj>
r"""°°°
## Machine Learning

Here we are going to create the different datasets that will be used throughout
the machine learning process. First, we will create a smaller sample of the full
dataset to allow faster execution of some machine learning models during development
and testing.

We will also generate an aggregated dataset that summarizes the review results
for each hotel, providing useful features such as average sentiment score or total
number of reviews.

After preparing these datasets, we will proceed to build and evaluate different
machine learning models for sentiment classification.

It is important to note that during the machine learning modeling process,
cross-validation was not applied. We are aware that using cross-validation is the
proper and more reliable approach for evaluating models, especially when combined
with appropriate data splitting. Although a train-test split was used, running
cross-validation proved to be quite time-consuming due to the size of the dataset
and the complexity of some models.

**Therefore, cross-validation may be applied in future steps, particularly for
models with longer training times or in order to further optimize performance.**
°°°"""
# |%%--%%| <zF6H9fHwAj|gWVGi1cWd7>
r"""°°°
Here we are creating a reduced version of the dataset by selecting the hotels
that have the most reviews, until we reach around 50,000 total rows.

First, we count how many reviews each hotel has and then we sort them in descending order.
°°°"""
# |%%--%%| <gWVGi1cWd7|sk2NrOIxk8>
r"""°°°
The main reason we created this reduced dataset is that some machine learning algorithms
take too much time to run on the full dataset. So, in order to speed up the process,
we decided to keep only the hotels with the most reviews until we reach around 50,000 rows.

We know that selecting hotels based on the number of reviews is a convenient approach
and not always realistic in practice, but it helps us develop and test our models
more efficiently during the early stages.

°°°"""
# |%%--%%| <sk2NrOIxk8|JHqRvkmaaN>

hotel_counts = cleaned.groupBy("Hotel_Name").agg(count("*").alias("review_count"))

# Step 2: Order hotels by number of reviews (descending)
ordered_hotels = hotel_counts.orderBy("review_count", ascending=False)

# Step 3: Collect hotel names and build cumulative total until ~50,000 rows
hotel_list = ordered_hotels.collect()

selected_hotels = []
running_total = 0

for row in hotel_list:
    hotel_name = row["Hotel_Name"]
    count_reviews = row["review_count"]
    if running_total + count_reviews <= 50000:
        selected_hotels.append(hotel_name)
        running_total += count_reviews
    else:
        break

clean_reduced_hotels_most_reviews = cleaned.filter(col("Hotel_Name").isin(selected_hotels))

# |%%--%%| <JHqRvkmaaN|8Tv87DcyJ1>
r"""°°°
Great — you want to create a new dataset that, for each hotel, contains the average values of selected numerical columns from your full dataset cleaned.


Step-by-step goal:
From cleaned, compute average values per hotel for the following columns:
°°°"""
# |%%--%%| <8Tv87DcyJ1|eqnNRlrm5u>

# Step 1: Compute averages
columns_to_average = [
    'Additional_Number_of_Scoring',
    'Average_Score',
    'Total_Number_of_Reviews',
    'Total_Number_of_Reviews_Reviewer_Has_Given',
    'days_since_review',
    'lat',
    'lng',
    'Num_Tags',
    'Positive_Lemma_Count',
    'Negative_Lemma_Count'
]

agg_exprs = [avg(col).alias(f"avg_{col}") for col in columns_to_average]

hotel_avg_stats = cleaned.groupBy("Hotel_Name").agg(*agg_exprs)

# Step 2: Compute review counts per hotel
hotel_counts = cleaned.groupBy("Hotel_Name").agg(count("*").alias("review_count"))

# Step 3: Join the two DataFrames
hotel_avg_stats_with_counts = hotel_avg_stats.join(hotel_counts, on="Hotel_Name")

# Step 4: Filter for hotels with more than 30 reviews
hotel_avg_stats_filtered = hotel_avg_stats_with_counts.filter("review_count > 30")

# Step 5: (Optional) Show result
hotel_avg_stats_filtered.show(truncate=False)


# |%%--%%| <eqnNRlrm5u|1QpabtuUGF>

hotel_avg_stats_filtered.count()


# |%%--%%| <1QpabtuUGF|w5tFqCymof>

# Print hotels that have missing coordinates
hotel_avg_stats_filtered.filter("avg_lat IS NULL OR avg_lng IS NULL").select("Hotel_Name", "avg_lat", "avg_lng").show(truncate=False)

# |%%--%%| <w5tFqCymof|D9z76Okohb>

# Remove those hotels from the dataset
hotel_avg_stats_filtered = hotel_avg_stats_filtered.filter("avg_lat IS NOT NULL AND avg_lng IS NOT NULL")

# |%%--%%| <D9z76Okohb|RkTFskNW6W>

print(f"Final number of hotels (after removing those with missing lat/lng): {hotel_avg_stats_filtered.count()}")


# |%%--%%| <RkTFskNW6W|2UOtJXDaL8>
r"""°°°
Here we build a regression model to predict the average hotel review score based
on aggregated features derived from user reviews.

We begin by computing per-hotel averages of various numerical attributes
(such as number of reviews, geographic coordinates, and word counts). We then filter
out hotels with fewer than 30 reviews and those missing geographic data.
Using these cleaned and aggregated values, we assemble a feature vector for each
hotel and train a linear regression model to estimate the average review score.

The dataset is split into training and testing sets, and we evaluate the model's
performance using RMSE and R² metrics to assess prediction accuracy and fit quality.

°°°"""
# |%%--%%| <2UOtJXDaL8|6NYnx82RuP>

# Step 1: Compute averages per hotel
columns_to_average = [
    'Additional_Number_of_Scoring',
    'Average_Score',
    'Total_Number_of_Reviews',
    'Total_Number_of_Reviews_Reviewer_Has_Given',
    'days_since_review',
    'lat',
    'lng',
    'Num_Tags',
    'Positive_Lemma_Count',
    'Negative_Lemma_Count'
]

agg_exprs = [avg(col).alias(f"avg_{col}") for col in columns_to_average]
hotel_avg_stats = cleaned.groupBy("Hotel_Name").agg(*agg_exprs)

# Step 2: Add review count and filter hotels with > 30 reviews
hotel_counts = cleaned.groupBy("Hotel_Name").agg(count("*").alias("review_count"))
hotel_avg_stats_with_counts = hotel_avg_stats.join(hotel_counts, on="Hotel_Name")
hotel_avg_stats_filtered = hotel_avg_stats_with_counts.filter("review_count > 30")

# Step 3: Remove hotels with null lat/lng
hotel_avg_stats_filtered = hotel_avg_stats_filtered.filter("avg_lat IS NOT NULL AND avg_lng IS NOT NULL")

# Step 4: Assemble features
feature_cols = [
    'avg_Additional_Number_of_Scoring',
    'avg_Total_Number_of_Reviews',
    'avg_Total_Number_of_Reviews_Reviewer_Has_Given',
    'avg_days_since_review',
    'avg_lat',
    'avg_lng',
    'avg_Num_Tags',
    'avg_Positive_Lemma_Count',
    'avg_Negative_Lemma_Count'
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_data = assembler.transform(hotel_avg_stats_filtered)

# Step 5: Prepare final dataset with label
final_data = assembled_data.select("features", col("avg_Average_Score").alias("label"))

# Step 6: Train-test split
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

# Step 7: Train linear regression model
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# Step 8: Make predictions on test set
predictions = lr_model.transform(test_data)

# Step 9: Evaluate model
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

print(f"RMSE on test set: {rmse}")
print(f"R² on test set: {r2}")


# |%%--%%| <6NYnx82RuP|CHvWPBfEMR>
r"""°°°
We can see that the linear regression model achieved an RMSE of approximately 0.38,
indicating that, on average, the predicted hotel scores deviate from the actual score
s by less than half a point. Additionally, the model achieved an R² of 0.49,
meaning it explains about 49% of the variance in average hotel scores.

This suggests that while the model captures some meaningful patterns in the data,
there is still substantial variability in hotel ratings that remains unexplained
by the current features. These results indicate a moderate predictive performance,
and could potentially be improved by using more complex models (like Random Forest
or Gradient Boosting) or by incorporating richer features such as text sentiment
from reviews or location-based attributes.
°°°"""
# |%%--%%| <CHvWPBfEMR|5wjsTjsDoT>

from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Step 1: Define models
models = {
    "Linear Regression": LinearRegression(featuresCol="features", labelCol="label"),
    "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=100),
    "Gradient Boosted Trees": GBTRegressor(featuresCol="features", labelCol="label", maxIter=100)
}

# Step 2: Train, predict, evaluate
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

results = []

for name, model in models.items():
    fitted = model.fit(train_data)
    preds = fitted.transform(test_data)

    rmse = evaluator_rmse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)

    results.append((name, rmse, r2))

# Step 3: Print results
for name, rmse, r2 in results:
    print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")


# |%%--%%| <5wjsTjsDoT|EXHVgEgs4E>
r"""°°°
Linear Regression performs best among the three models — it has the lowest RMSE (0.3766)
and the highest R² (0.4892). This suggests that, despite being a simpler model,
it captures the underlying patterns in your aggregated dataset more effectively
than the tree-based models.

Random Forest and GBT perform slightly worse, with higher RMSE and lower R².
This may be surprising since these models typically outperform linear models on
complex data — but in your case, the features are already averaged per hotel,
which may remove much of the non-linear variation and noise that tree models are
good at exploiting.

This implies your data relationships are relatively linear, or at least
well-approximated by a linear function at this aggregation level.
°°°"""
# |%%--%%| <EXHVgEgs4E|hLAiSK08Mh>

pdf = preds.select("prediction", "label").toPandas()
plt.scatter(pdf["label"], pdf["prediction"], alpha=0.3)
plt.plot([pdf["label"].min(), pdf["label"].max()], [pdf["label"].min(), pdf["label"].max()], color='red')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"{name} - Actual vs. Predicted")
plt.show()


# |%%--%%| <hLAiSK08Mh|lBJNeEsFhX>

if hasattr(fitted, "featureImportances"):
    importances = fitted.featureImportances
    print(f"{name} feature importances:\n{importances}")

# |%%--%%| <lBJNeEsFhX|W4lWEQGhlS>

# Assume 'feature_cols' is your list of feature column names
feature_cols = [...]  # <- replace with your actual feature names

for name, model in models.items():
    fitted = model.fit(train_data)
    preds = fitted.transform(test_data)

    rmse = evaluator_rmse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)

    print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Feature importances (for tree-based models)
    if hasattr(fitted, "featureImportances"):
        importances = fitted.featureImportances
        importance_list = importances.toArray()

        print("Feature Importances:")
        for feat, score in zip(feature_cols, importance_list):
            print(f"  {feat}: {score:.4f}")

# |%%--%%| <W4lWEQGhlS|sTqfuRsZZ2>
r"""°°°
### Model 3
°°°"""
# |%%--%%| <sTqfuRsZZ2|LAxVurgFh3>

clean_reduced_hotels_most_reviews.filter(
    (col("lat").isNull()) | (col("lng").isNull())
).select("Hotel_Name", "lat", "lng").show(truncate=False)

# |%%--%%| <LAxVurgFh3|f9m3Dy14SP>

hotel_locations = clean_reduced_hotels_most_reviews \
    .select("Hotel_Name", "lat", "lng") \
    .dropna(subset=["lat", "lng"]) \
    .dropDuplicates(["Hotel_Name"])

# |%%--%%| <f9m3Dy14SP|YyGDi7y3Kp>

hotel_locations.count()

# |%%--%%| <YyGDi7y3Kp|AQN1w2nZ6M>

hotel_coords = hotel_locations.select("Hotel_Name", "lat", "lng").collect()

# |%%--%%| <AQN1w2nZ6M|xySrlD9GvB>

from geopy.geocoders import Nominatim
from time import sleep

geolocator = Nominatim(user_agent="geoapi")

results = []
for row in hotel_coords:
    hotel_name, lat, lng = row["Hotel_Name"], row["lat"], row["lng"]
    try:
        location = geolocator.reverse((lat, lng), exactly_one=True)
        address = location.raw["address"]
        city = address.get("city", address.get("town", address.get("village", "")))
        country = address.get("country", "")
        results.append((hotel_name, city, country))
    except Exception as e:
        results.append((hotel_name, None, None))
    sleep(1)  # avoid rate limiting

# Convert results to a DataFrame or export


# |%%--%%| <xySrlD9GvB|H9jgRBngDP>

for r in results:
    print(r)


# |%%--%%| <H9jgRBngDP|CvlScRDSI5>

import pandas as pd

geo_df = pd.DataFrame(results, columns=["Hotel_Name", "City", "Country"])
print(geo_df.head())

# |%%--%%| <CvlScRDSI5|csYybb9Fw1>
r"""°°°
We attempted to use the same code in order to retrieve the country and city
information for all hotels.

However, we encountered issues due to limitations of the API service (Nominatim).
Despite trying to adjust the timeout settings and add delays between requests,
the process proved to be very time-consuming.

For this reason, we applied the method only to a sample of 17 hotels.

°°°"""
# |%%--%%| <csYybb9Fw1|xATckEweYf>

clean_reduced_hotels_most_reviews.columns


# |%%--%%| <xATckEweYf|rWV690OQcU>

from pyspark.sql.functions import size

empty_counts = clean_reduced_hotels_most_reviews.select(
    (size(col("Negative_Lemmatized")) == 0).alias("Negative_Lemmatized_empty"),
    (size(col("Positive_Lemmatized")) == 0).alias("Positive_Lemmatized_empty")
).groupBy().sum()

empty_counts.show()

# |%%--%%| <rWV690OQcU|aFPmARxtkN>

from pyspark.sql.functions import col, sum

clean_reduced_hotels_most_reviews.select(
    sum(col("Negative_Lemmatized").isNull().cast("int")).alias("null_Negative_Lemmatized"),
    sum(col("Positive_Lemmatized").isNull().cast("int")).alias("null_Positive_Lemmatized")
).show()


# |%%--%%| <aFPmARxtkN|mzSFBwzT9z>

null_or_empty = clean_reduced_hotels_most_reviews.filter(
    col("Negative_Lemmatized").isNull() | (size(col("Negative_Lemmatized")) == 0) |
    col("Positive_Lemmatized").isNull() | (size(col("Positive_Lemmatized")) == 0)
)

print(f"Rows with null or empty lemmatized tokens: {null_or_empty.count()}")


# |%%--%%| <mzSFBwzT9z|OQbY7KqkgW>

# Count rows where Negative_Lemmatized is an empty array
empty_negative = clean_reduced_hotels_most_reviews.filter(size(col("Negative_Lemmatized")) == 0).count()
print(f"Rows with empty Negative_Lemmatized: {empty_negative}")

# Count rows where Positive_Lemmatized is an empty array
empty_positive = clean_reduced_hotels_most_reviews.filter(size(col("Positive_Lemmatized")) == 0).count()
print(f"Rows with empty Positive_Lemmatized: {empty_positive}")

# Optional: Count rows where both are empty
both_empty = clean_reduced_hotels_most_reviews.filter(
    (size(col("Negative_Lemmatized")) == 0) & (size(col("Positive_Lemmatized")) == 0)
).count()
print(f"Rows where BOTH are empty: {both_empty}")


# |%%--%%| <OQbY7KqkgW|Jm5S6Rcz5i>
clean_reduced_hotels_most_reviews.show()

# |%%--%%| <Jm5S6Rcz5i|QhRDemnwk9>
r"""°°°
In this approach, we aimed to separate the positive and negative components of each
hotel review so they could be treated as independent training samples.

To achieve this, we filtered out rows where either the positive or negative lemmatized
arrays were non-empty. We then created two separate DataFrames — one containing
only the positive reviews (labeled as 1) and another containing only the negative
reviews (labeled as 0) — and combined them into a single dataset. This allowed us
to isolate each sentiment and increase the volume of meaningful training data.

We proceeded to build a machine learning pipeline using PySpark, applying HashingTF
and IDF for feature extraction and training a logistic regression model.

Finally, we evaluated the model’s performance using the AUC metric to assess its
ability to distinguish between positive and negative sentiment.
°°°"""
# |%%--%%| <QhRDemnwk9|4ZqUBJva0o>

# Filter rows that have non-empty lemmatized arrays
df = clean_reduced_hotels_most_reviews.filter(
    (size(col("Positive_Lemmatized")) > 0) | (size(col("Negative_Lemmatized")) > 0)
)

# Create one DataFrame for positive reviews
positive_df = df.filter(size(col("Positive_Lemmatized")) > 0) \
    .withColumn("lemmatized_tokens", col("Positive_Lemmatized")) \
    .withColumn("label", lit(1)) \
    .select("lemmatized_tokens", "label")

# Create one DataFrame for negative reviews
negative_df = df.filter(size(col("Negative_Lemmatized")) > 0) \
    .withColumn("lemmatized_tokens", col("Negative_Lemmatized")) \
    .withColumn("label", lit(0)) \
    .select("lemmatized_tokens", "label")

# Combine both
df = positive_df.union(negative_df)

# Select only the necessary columns for training
df = df.select("lemmatized_tokens", "label")

# Define the ML pipeline stages
hashing_tf = HashingTF(inputCol="lemmatized_tokens", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[hashing_tf, idf, lr])

# Split into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_data)

# Make predictions and evaluate
predictions = model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)

print(f"AUC on test set: {auc}")

# Show predictions
predictions.select("lemmatized_tokens", "label", "prediction").show(5, truncate=False)

# |%%--%%| <4ZqUBJva0o|g88tERS2Fk>

df.groupBy("label").count().show()

# |%%--%%| <g88tERS2Fk|Pi1t2FA3gK>

# TODO: This looks like a duplicate

# Prepare positive and negative samples

# Filter non-empty reviews
pos_df = clean_reduced_hotels_most_reviews.filter(lower(col("Positive_Review")) != "no positive") \
    .withColumn("text", col("Positive_Review")) \
    .withColumn("label", lit(1)) \
    .select("text", "label")

neg_df = clean_reduced_hotels_most_reviews.filter(lower(col("Negative_Review")) != "no negative") \
    .withColumn("text", col("Negative_Review")) \
    .withColumn("label", lit(0)) \
    .select("text", "label")

# Combine both into a single DataFrame
df = pos_df.union(neg_df)

# Define the text processing pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, lr])

# Split into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_data)

# Evaluate
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)

print(f"AUC on test set: {auc}")

# Show some predictions
predictions.select("text", "label", "prediction").show(5, truncate=False)

# |%%--%%| <Pi1t2FA3gK|mR2ScC4orh>

# TODO: Why are we adding new text? Justify and maybe add to new section

new_texts = [
    "The staff was incredibly helpful and the room was very clean.",
    "We waited an hour for check-in and the bathroom was disgusting."
]

# Create DataFrame
new_df = spark.createDataFrame([(t,) for t in new_texts], ["text"])

# Predict
model.transform(new_df).select("text", "prediction").show(truncate=False)


# |%%--%%| <mR2ScC4orh|YIGu2QAV6w>

df.groupBy("label").count().show()


# |%%--%%| <YIGu2QAV6w|V2m2sPHRPt>

# Get predictions and actual labels
results = predictions.select("label", "prediction")

# Calculate confusion matrix components
tp = results.filter((col("label") == 1) & (col("prediction") == 1)).count()
tn = results.filter((col("label") == 0) & (col("prediction") == 0)).count()
fp = results.filter((col("label") == 0) & (col("prediction") == 1)).count()
fn = results.filter((col("label") == 1) & (col("prediction") == 0)).count()

print(f"Confusion Matrix:")
print(f"True Positives:  {tp}")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

# |%%--%%| <V2m2sPHRPt|QzuNyoskHe>

for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
    print(f"{metric.capitalize()}: {evaluator.evaluate(predictions):.4f}")

# |%%--%%| <QzuNyoskHe|JKumUXvhZW>

new_texts = [
    "The room was clean and quiet. Excellent service!",
    "It was a terrible experience, dirty bathroom and rude staff.",
    "Great value for the price.",
    "The air conditioning didn’t work, and it was very noisy.",
    "Check-in was fast and smooth.",
    "No towels in the room and the shower was broken."
]

# Create DataFrame
new_df = spark.createDataFrame([(t,) for t in new_texts], ["text"])

# Predict
predictions = model.transform(new_df)

# Show predicted label and probability
predictions.select("text", "prediction", "probability").show(truncate=False)

# |%%--%%| <JKumUXvhZW|qIC9fuCYlt>
r"""°°°
 Adding bigrams (n-grams) to capture short phrases like:
"no towels"

"was broken"

"not clean"

These phrases are more meaningful than isolated words like "no" or "towels" on their own.


°°°"""
# |%%--%%| <qIC9fuCYlt|hTySuUS0K2>

# Prepare positive and negative samples

# Filter out empty placeholder reviews
pos_df = cleaned.filter(lower(col("Positive_Review")) != "no positive") \
    .withColumn("text", col("Positive_Review")) \
    .withColumn("label", lit(1)) \
    .select("text", "label")

neg_df = cleaned.filter(lower(col("Negative_Review")) != "no negative") \
    .withColumn("text", col("Negative_Review")) \
    .withColumn("label", lit(0)) \
    .select("text", "label")

# Combine positive and negative into one DataFrame
df = pos_df.union(neg_df)

# Define the text processing pipeline with bigrams
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
bigrams = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams")
hashing_tf = HashingTF(inputCol="bigrams", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, remover, bigrams, hashing_tf, idf, lr])

# Train/test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train the model
model = pipeline.fit(train_data)

# Evaluate the model
predictions = model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)
print(f"AUC on test set: {auc}")

# View sample predictions with probability
predictions.select("text", "label", "prediction", "probability").show(5, truncate=False)

# Confusion matrix (basic form)
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# |%%--%%| <hTySuUS0K2|Rincg7DGyl>

# New review examples
new_texts = [
    "No towels in the room and the shower was broken.",
    "Everything was clean and comfortable.",
    "Staff were rude and unhelpful.",
    "Check-in was fast and easy.",
    "The hotel was noisy and poorly maintained."
]

# Create DataFrame for new inputs
new_df = spark.createDataFrame([(t,) for t in new_texts], ["text"])

# Predict
predicted_new = model.transform(new_df)
predicted_new.select("text", "prediction", "probability").show(truncate=False)

# |%%--%%| <Rincg7DGyl|l0OqeSn4oX>

df.groupBy("label").count().orderBy("label").show()


# |%%--%%| <l0OqeSn4oX|NsqAR92lLq>
r"""°°°
Option 1: Combine unigrams + bigrams
This helps the model learn both:

Individual words like "clean", "rude", "broken"

Phrases like "no towels", "was broken"
°°°"""
# |%%--%%| <NsqAR92lLq|oJR7sfPLSq>

# Tokenize and remove stopwords
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

# Create unigrams and bigrams
unigrams_tf = HashingTF(inputCol="filtered_tokens", outputCol="uni_raw", numFeatures=5000)
bigrams = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams")
bigrams_tf = HashingTF(inputCol="bigrams", outputCol="bi_raw", numFeatures=5000)

# Combine unigrams and bigrams features
assembler = VectorAssembler(inputCols=["uni_raw", "bi_raw"], outputCol="raw_features")

# Apply IDF
idf = IDF(inputCol="raw_features", outputCol="features")

# Classifier
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Build pipeline
pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    unigrams_tf,
    bigrams,
    bigrams_tf,
    assembler,
    idf,
    lr
])

# |%%--%%| <oJR7sfPLSq|CgfRc5fyD2>

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)
print(f"AUC (Unigrams + Bigrams): {auc}")

# |%%--%%| <CgfRc5fyD2|Gi2xCvrrnM>

df.groupBy("label").count().show()

# |%%--%%| <Gi2xCvrrnM|alKv12MNAc>

evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel")

f1 = evaluator_f1.evaluate(predictions)
precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)

print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# |%%--%%| <alKv12MNAc|3uuJOizROk>

from IPython import display

evaluation_summary = """
### Evaluation Summary

| Metric     | Value  | Interpretation                                              |
|------------|--------|-------------------------------------------------------------|
| **AUC**    | 0.973  | Excellent. Your model ranks positive > negative reliably.   |
| **F1 Score** | 0.922 | Strong overall balance between precision and recall.       |
| **Precision** | 0.904 | 90.4% of predicted positives were correct (low false positives). |
| **Recall** | 0.915  | 91.5% of actual positives were detected (low false negatives). |
"""

display(display.Markdown(evaluation_summary))

# |%%--%%| <3uuJOizROk|mE9kiS7tJU>
r"""°°°
Confusion Matrix
°°°"""
# |%%--%%| <mE9kiS7tJU|bHLTmVvKU5>

# Assuming `predictions` is the DataFrame after model.transform()

predictions.select("label", "prediction") \
    .groupBy("label", "prediction") \
    .count() \
    .orderBy("label", "prediction") \
    .show()


# |%%--%%| <bHLTmVvKU5|grVPsgG6os>

# Example new reviews to classify
new_texts = [
    "The room was spotless and the staff were incredibly friendly.",
    "There was no hot water and the heater was broken.",
    "Excellent location and fast check-in process.",
    "Unhelpful reception and dirty sheets.",
    "Everything was perfect! I would stay again.",
    "No towels in the room and the bathroom smelled terrible."
]

# Create DataFrame from new inputs
new_df = spark.createDataFrame([(text,) for text in new_texts], ["text"])

# Use your trained pipeline model to predict
new_predictions = model.transform(new_df)

# Show predicted label and probability
new_predictions.select("text", "prediction", "probability").show(truncate=False)

# |%%--%%| <grVPsgG6os|4tVQrJJkbG>
r"""°°°
### MODEL 4
°°°"""
# |%%--%%| <4tVQrJJkbG|ZI24XwjvJH>

hotel_avg_stats_filtered.show()

# |%%--%%| <ZI24XwjvJH|Ajjt0gBVk4>

# Sort hotels by average number of reviews in ascending order
hotel_avg_stats_filtered.orderBy("avg_Total_Number_of_Reviews", ascending=True).show()

# |%%--%%| <Ajjt0gBVk4|lopB5kVTUH>
r"""°°°
Here we create a new column called `star_rating`, where we convert the average score
of each hotel into a 1–5 star scale.

This helps us simplify the target variable for classification tasks.
°°°"""
# |%%--%%| <lopB5kVTUH|JXPTCt0M6q>

# Here we add a new column that assigns a star rating (1 to 5) based on the avg_Average_Score of each hotel
hotel_with_stars = hotel_avg_stats_filtered.withColumn(
    "star_rating",
    when(col("avg_Average_Score") >= 9, 5)
    .when(col("avg_Average_Score") >= 8, 4)
    .when(col("avg_Average_Score") >= 6, 3)
    .when(col("avg_Average_Score") >= 5, 2)
    .otherwise(1)
)

# |%%--%%| <JXPTCt0M6q|s46x0T0xAm>

feature_columns = [
    "avg_Additional_Number_of_Scoring",
    "avg_Total_Number_of_Reviews",
    "avg_Total_Number_of_Reviews_Reviewer_Has_Given",
    "avg_days_since_review",
    "avg_lat",
    "avg_lng",
    "avg_Num_Tags",
    "avg_Positive_Lemma_Count",
    "avg_Negative_Lemma_Count",
    "review_count"
]

# |%%--%%| <s46x0T0xAm|Pgtnhh6luM>

# Split the data into training and test sets
train_data, test_data = hotel_with_stars.randomSplit([0.8, 0.2], seed=42)

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Create a classifier
rf = RandomForestClassifier(labelCol="star_rating", featuresCol="features", numTrees=100)

# Create pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Fit model on training data
model = pipeline.fit(train_data)

# Make predictions on test data
predictions = model.transform(test_data)


# |%%--%%| <Pgtnhh6luM|lCPAAKMTmI>

# Show actual vs predicted star ratings side by side
predictions.select("Hotel_Name", "avg_Average_Score", "star_rating", "prediction").show()

# |%%--%%| <lCPAAKMTmI|FWNZ4gRXAS>

# Filter and show misclassified hotels
misclassified = predictions.filter(col("star_rating") != col("prediction"))
misclassified.select("Hotel_Name", "avg_Average_Score", "star_rating", "prediction").show()

# |%%--%%| <FWNZ4gRXAS|cycRq5ilkP>

# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="star_rating", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator_acc.evaluate(predictions)

# F1-score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="star_rating", predictionCol="prediction", metricName="f1"
)
f1 = evaluator_f1.evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# |%%--%%| <cycRq5ilkP|vX7WfvgiQ0>

# Confusion matrix (counts of actual vs predicted)
confusion_matrix = predictions.groupBy("star_rating", "prediction").count().orderBy("star_rating", "prediction")
confusion_matrix.show()

# |%%--%%| <vX7WfvgiQ0|cCVlT0E6so>

# Group by actual and predicted values and convert to pandas
confusion_pd = predictions.groupBy("star_rating", "prediction").count().toPandas()

# |%%--%%| <cCVlT0E6so|GiaPLdzzM0>

# Pivot the confusion matrix for plotting
conf_matrix = confusion_pd.pivot(index="star_rating", columns="prediction", values="count").fillna(0)

# |%%--%%| <GiaPLdzzM0|0OXjiDUbJq>

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="Blues")
plt.title("Confusion Matrix (Actual vs Predicted Star Ratings)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# |%%--%%| <0OXjiDUbJq|AvvpHFhvCN>

# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(labelCol="star_rating", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(predictions)

# F1 Score
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="star_rating", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# |%%--%%| <AvvpHFhvCN|hb1y1R2Dv6>
r"""°°°
Since we are going to build classification models to predict hotel star ratings,
we first check the distribution of the `star_rating` column.

This helps us understand whether the classes are balanced or imbalanced.
If some star ratings appear much more frequently than others, it could affect the
performance of our models, especially in terms of fairness and accuracy across classes.

°°°"""
# |%%--%%| <hb1y1R2Dv6|6v31mzFQAp>

# we count how many hotels belong to each star rating so we can check if the data is balanced
hotel_with_stars.groupBy("star_rating").count().orderBy("star_rating").show()

# |%%--%%| <6v31mzFQAp|a7Sn7HRl8e>
r"""°°°
We can say that the dataset is heavily skewed toward higher-rated hotels,
especially 4-star ones. This imbalance might affect model performance if we're
analyzing or predicting based on star rating, as the underrepresented classes (like 2-star)
may not provide enough data to learn from effectively.
°°°"""
# |%%--%%| <a7Sn7HRl8e|e6xgJI2Bez>
r"""°°°
Initially, we will begin building the model without addressing the class imbalance
in hotel star ratings. This approach will give us a baseline understanding of the
model's performance under real-world conditions, where higher-rated hotels (e.g., 4-star)
dominate the dataset.

However, we will later apply balancing techniques such as resampling, SMOTE, or
class weighting, because we anticipate the following issues:
°°°"""
# |%%--%%| <e6xgJI2Bez|9XV7r3aiHc>
# we shift the star ratings to start from 0 so they work with models like Naive Bayes and MLP
hotel_adjusted = hotel_with_stars.withColumn("star_rating_adj", col("star_rating") - 1)

# we split our dataset into train and test sets (80% train, 20% test)
train_data, test_data = hotel_adjusted.randomSplit([0.8, 0.2], seed=42)

# we use the same assembler to turn our features into a single feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# here we define the models we want to try out
lr = LogisticRegression(labelCol="star_rating_adj", featuresCol="features", maxIter=100)
dt = DecisionTreeClassifier(labelCol="star_rating_adj", featuresCol="features")
rf = RandomForestClassifier(labelCol="star_rating_adj", featuresCol="features", numTrees=100)
mlp = MultilayerPerceptronClassifier(
    labelCol="star_rating_adj",
    featuresCol="features",
    layers=[len(feature_columns), 10, 5],  # we assume we have 5 different star ratings (0–4)
    maxIter=100
)

# we set up evaluators to calculate accuracy and F1 score
evaluator_acc = MulticlassClassificationEvaluator(labelCol="star_rating_adj", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="star_rating_adj", predictionCol="prediction", metricName="f1")

# function where we train the model, make predictions, and print the results
def evaluate_model(name, classifier):
    pipeline = Pipeline(stages=[assembler, classifier])
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)

    acc = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print(f"{name:<25} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    return predictions

# now we run all the models and see how they perform
evaluate_model("Logistic Regression", lr)
evaluate_model("Decision Tree", dt)
evaluate_model("Random Forest", rf)
evaluate_model("Multilayer Perceptron", mlp)


# |%%--%%| <9XV7r3aiHc|Rb3fN85vfz>
r"""°°°
Here we can see the performance of different classification models on the task of
predicting hotel star ratings.

Logistic Regression achieved the best results, with an accuracy of 71.25% and an
F1 Score of 69.85%, showing a good balance between correct predictions and overall
model consistency.

The Decision Tree model followed closely, reaching 70.42% accuracy
and a 67.90% F1 Score, which indicates decent performance but slightly less robust
than Logistic Regression.

Random Forest performed worse than expected, with an accuracy of 69.58% and a lower
F1 Score of 64.67%, possibly due to overfitting or sensitivity to the class distribution.

The Multilayer Perceptron had the lowest scores, achieving 65.83% accuracy and just
52.27% F1 Score, suggesting that the model struggled to generalize effectively across
the multiple classes.

Overall, Logistic Regression appears to be the most suitable model for this dataset
in its current form.

°°°"""
# |%%--%%| <Rb3fN85vfz|EepiOHODFA>
r"""°°°
### Clustering
°°°"""
# |%%--%%| <EepiOHODFA|61oAK1WM9e>

hotel_with_stars.count()

# |%%--%%| <61oAK1WM9e|Lr6Tjmi2Et>

hotel_with_stars.show()

# |%%--%%| <Lr6Tjmi2Et|FgxVbBx04b>

cleaned.count()

# |%%--%%| <FgxVbBx04b|KZ6SzGKb0K>

cleaned.show()

# |%%--%%| <KZ6SzGKb0K|UOG12irSOp>
r"""°°°
Here we use K-Means clustering to group hotels based on review-related features
such as average score, sentiment counts, and review activity.

We then compare the resulting clusters with the actual star ratings to see if
similar hotels tend to share the same rating.
°°°"""
# |%%--%%| <UOG12irSOp|PPffkthyu0>
r"""°°°
We are using the additional dataset we previously created with star ratings
through the machine learning model.
°°°"""
# |%%--%%| <PPffkthyu0|AOLNvYkzwJ>

# we choose the features we think are most relevant for grouping the hotels based on guest reviews and behavior
selected_features = [
    "avg_Average_Score",
    "avg_Positive_Lemma_Count",
    "avg_Negative_Lemma_Count",
    "avg_Num_Tags",
    "avg_Total_Number_of_Reviews",
    "avg_days_since_review"
]

# we turn the selected columns into a single feature vector so we can use it in the KMeans algorithm
assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
cluster_input = assembler.transform(hotel_with_stars)

# we apply KMeans to split the hotels into 3 groups based on the features we picked
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=3, seed=42)
model = kmeans.fit(cluster_input)
clustered = model.transform(cluster_input)

# we now look at how the actual star ratings are spread across the different clusters
clustered.groupBy("cluster", "star_rating").count().orderBy("cluster", "star_rating").show()


# |%%--%%| <AOLNvYkzwJ|0ZU0m76OK2>
r"""°°°
The table shows how many hotels in each cluster (0, 1, or 2) belong to each star
rating category (from 2 to 5).

This helps us understand whether the clusters formed by K-Means align with the
actual hotel quality.

Cluster 0 is clearly the largest and is mainly made up of 4-star hotels (622)
and a significant number of 5-star hotels (244), with fewer 3-star (143) and
just one 2-star hotel.

This suggests that Cluster 0 likely represents well-rated or popular hotels with
strong guest feedback. Cluster 1 is much smaller, with most hotels being 4-star (41),
followed by 3-star (13) and just a few 5-star hotels (5).

It possibly represents average-performing or low-volume hotels with less consistent
review patterns. Cluster 2 mostly contains 4-star hotels (234), along with some
3-star (60) and 5-star hotels (34), which might indicate solid mid- to high-performing
hotels with slightly different review characteristics than those in Cluster 0.

Overall, the clustering captures differences in hotel quality to some extent,
with Cluster 0 standing out as the group with the highest-rated hotels.
°°°"""
# |%%--%%| <0ZU0m76OK2|q9lQWJy6aj>
r"""°°°
To improve the clustering and better understand how well our data is grouped,
we decided to test different values for k (the number of clusters).

This is an important step in clustering, since choosing the right number of clusters
can reveal more meaningful patterns.

In PySpark, we can evaluate this by checking the Within Set Sum of Squared Errors (WSS),
also known as the cost or inertia.

By calculating the WSS for different values of k (for example from 2 to 8),
we can use the elbow method to find the point where increasing the number of clusters
stops giving us significantly better results. That point usually indicates the
most appropriate number of clusters.
°°°"""
# |%%--%%| <q9lQWJy6aj|ciZ9LvWoVG>
r"""°°°
Here we test different values for k to figure out which number of clusters works best.
We use the Within Set Sum of Squared Errors (WSS) to evaluate each model and look
for the point where the error stops dropping significantly — that helps us decide
the optimal number of clusters.

°°°"""
# |%%--%%| <ciZ9LvWoVG|e0iXO2BSRA>

# we test different values for k to find the best number of clusters
for k in range(2, 9):
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42)
    model = kmeans.fit(cluster_input)
    wss = model.summary.trainingCost
    print(f"k = {k}, Within Set Sum of Squared Errors = {wss:.2f}")


# |%%--%%| <e0iXO2BSRA|k21qJYhZFI>
r"""°°°
We tested different values of k and saw that the WSS drops significantly until
around k = 4 or 5, suggesting that these are good options for clustering.
Based on this, we now continue with k = 4 to see how the clusters relate to star ratings.

**Next, we’ll analyze the average values in each cluster to understand what kind
of hotels each group represents, and possibly use the cluster ID as a new feature
in our classification models.**

°°°"""
# |%%--%%| <k21qJYhZFI|PmJaowXW4O>
r"""°°°
TODO: Mention elbow method on plot to vizualize WDD drop as k increased

Since we saw that k = 4 offers a good balance between model simplicity and error
reduction, we use this value to apply KMeans clustering on the hotel datas

°°°"""
# |%%--%%| <PmJaowXW4O|w1SxUqvYA4>

# We start by selecting the features that we believe best represent hotel review behavior
selected_features = [
    "avg_Average_Score",
    "avg_Positive_Lemma_Count",
    "avg_Negative_Lemma_Count",
    "avg_Num_Tags",
    "avg_Total_Number_of_Reviews",
    "avg_days_since_review"
]

# We combine the selected features into a single vector to prepare the data for clustering
assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
cluster_input = assembler.transform(hotel_with_stars)

# Since we identified k = 4 as a good choice, we apply KMeans to group the hotels into 4 clusters
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=4, seed=42)
model = kmeans.fit(cluster_input)
clustered = model.transform(cluster_input)

# We check how the actual star ratings are distributed across the clusters
clustered.groupBy("cluster", "star_rating").count().orderBy("cluster", "star_rating").show()

# We also compute the average of each feature per cluster to understand what characterizes each group
clustered.groupBy("cluster").agg({col: "mean" for col in selected_features}).show()


# |%%--%%| <w1SxUqvYA4|H7YuWJgHtx>
r"""°°°
Our clustering results show that Cluster 0 is the largest and most diverse,
mainly composed of 4- and 5-star hotels, suggesting it's the core group of
well-rated hotels.

Clusters 1 and 2 also include mostly 4-star hotels but differ slightly in their
mix of 3- and 5-star ratings, indicating variations in review behavior or quality.

Cluster 3 is the smallest, with only 3- and 4-star hotels, possibly representing
a niche or lower-activity group.

Overall, the clusters align reasonably well with hotel star ratings.
°°°"""
# |%%--%%| <H7YuWJgHtx|8CXvDMOEUt>
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(cluster_input)
pca_result = pca_model.transform(cluster_input)

# Add cluster labels
pca_result = model.transform(pca_result)

# Convert to Pandas for plotting
pandas_df = pca_result.select("pca_features", "cluster").toPandas()
pandas_df[['PC1', 'PC2']] = pandas_df['pca_features'].apply(lambda x: pd.Series(x.toArray()))

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
for cluster_id in pandas_df['cluster'].unique():
    subset = pandas_df[pandas_df['cluster'] == cluster_id]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Cluster {cluster_id}')
plt.legend()
plt.title("PCA Projection of Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# |%%--%%| <8CXvDMOEUt|ox7tw5Q6TU>
r"""°°°
In this step, we use the ClusteringEvaluator to calculate the Silhouette Score,
which helps us assess the quality of our clusters.

We set the evaluator to use squared Euclidean distance and apply it to our clustered data.
The resulting score tells us how well each hotel fits within its assigned cluster,
and a higher value indicates better-defined and more separated clusters.

°°°"""
# |%%--%%| <ox7tw5Q6TU|nWqjk3ZQGx>

evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(clustered)
print(f"Silhouette Score: {silhouette:.3f}")

# |%%--%%| <nWqjk3ZQGx|6Bp42ZWUe9>
r"""°°°
To assess the quality of our clustering, we calculated the `Silhouette Score`,
which ranges from **-1 to 1**:
  - A score close to **+1** means the hotel is well-matched to its cluster and far from others.
  - A score around **0** indicates overlap between clusters.
  - A negative score suggests possible misclassification.



°°°"""
# |%%--%%| <6Bp42ZWUe9|T8BtJvnqrL>
r"""°°°
We evaluated the quality of our clusters using the Silhouette Score and obtained
a value of 0.754, which indicates that the clusters are well-defined and clearly separated.

This high score suggests that most hotels are strongly associated with their assigned
cluster and not overlapping with others, supporting our choice of using k = 4.

It also confirms that the features we selected provide meaningful structure for
grouping the hotels based on guest review behavior.
°°°"""
# |%%--%%| <T8BtJvnqrL|xG18lhe8Va>

from pyspark.ml.clustering import BisectingKMeans
bkm = BisectingKMeans(featuresCol="features", predictionCol="cluster", k=4)
bkm_model = bkm.fit(cluster_input)
bkm_clustered = bkm_model.transform(cluster_input)


# |%%--%%| <xG18lhe8Va|eF9FEYa8Ib>

bkm_clustered.groupBy("cluster", "star_rating").count().orderBy("cluster", "star_rating").show()


# |%%--%%| <eF9FEYa8Ib|bIZs5pdxJB>
r"""°°°
In this analysis, we grouped hotel reviews into four clusters and examined the
distribution of star ratings (3, 4, and 5 stars) within each cluster.

We observe that Cluster 0 contains the largest number of reviews, with a high
concentration of 4- and 5-star ratings, suggesting mostly positive feedback.

Cluster 1 also shows a predominance of 4-star reviews but has fewer 5-star ratings
compared to Cluster 0. Cluster 2 contains fewer reviews overall, with a noticeable
drop in 5-star ratings, indicating a potential shift toward more neutral or slightly
negative sentiment.

Cluster 3 has the smallest number of reviews and a low count across all star levels,
possibly representing outliers or a niche group of reviews.

This distribution helps us interpret the nature of each cluster and assess how
review sentiment varies across them.
°°°"""
# |%%--%%| <bIZs5pdxJB|tagiRGED5y>

evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette_bkm = evaluator.evaluate(bkm_clustered)
print(f"Silhouette Score (Bisecting KMeans): {silhouette_bkm:.3f}")


# |%%--%%| <tagiRGED5y|b5qw0YAu7d>
r"""°°°
That’s a great result — even slightly better than your previous KMeans score (which was 0.754).


°°°"""
# |%%--%%| <b5qw0YAu7d|sc0poqgfg6>
r"""°°°
We evaluated Bisecting KMeans using the Silhouette Score and obtained a value of 0.768,
which is slightly higher than the score we got with standard KMeans (0.754).

This indicates that Bisecting KMeans produced better-separated and more cohesive
clusters for our dataset.

It suggests that the hierarchical approach of Bisecting KMeans may be more suitable
for the structure of our hotel review features.

Given this, we could consider using the Bisecting KMeans clustering results as
our final version or at least give it more weight when interpreting the patterns
in hotel groupings.
°°°"""
# |%%--%%| <sc0poqgfg6|1IZKU07ZzB>
r"""°°°
#### Clustering Method Comparison

| Clustering Method    | Silhouette Score | Notes                                       |
|----------------------|------------------|---------------------------------------------|
| KMeans               | 0.754            | Good performance with well-separated groups |
| Bisecting KMeans     | 0.768            | Slightly better cluster cohesion and separation |


Based on the silhouette scores, Bisecting KMeans performed slightly better.
Both methods produced meaningful clusters, but the hierarchical approach of
Bisecting KMeans seems to fit our data structure more naturally.

In this analysis, we applied clustering techniques to group hotels based on guest
review behavior using features like average score, sentiment lemmas, and review count.
We tested both **KMeans** and **Bisecting KMeans** algorithms with `k=4`, based
on the elbow method and WSSSE analysis.

The **Silhouette Score** for KMeans was **0.754**, while Bisecting KMeans achieved
a slightly higher score of **0.768**, indicating better-defined clusters.

We also confirmed these groupings visually through PCA, and observed that clusters
aligned reasonably well with actual star ratings.

These clusters reveal different behavioral patterns among hotels, e.g., high-rated,
high-engagement vs. low-rated, low-activity groups, and can be useful for tasks
like **targeted marketing**, **service improvement**, or as a **feature** in future
predictive models.
°°°"""
# |%%--%%| <1IZKU07ZzB|D8hh8q4rLg>
r"""°°°
 Visualize Cluster Feature Averages with Bar Plot
°°°"""
# |%%--%%| <D8hh8q4rLg|kWUkcxpK5t>

# Get mean of each feature per cluster

means = bkm_clustered.groupBy("cluster").agg(
    *[mean(c).alias(c) for c in selected_features]
).toPandas().set_index("cluster")


# |%%--%%| <kWUkcxpK5t|wwEuT7nDL1>

# Normalize the feature means (Min-Max Scaling)
normalized = (means - means.min()) / (means.max() - means.min())

# Plot normalized data
normalized.T.plot(kind='bar', figsize=(10, 6), legend=True)
plt.title("Normalized Feature Values per Cluster (Bisecting KMeans)")
plt.ylabel("Normalized Value (0-1)")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# |%%--%%| <wwEuT7nDL1|egyFAue58i>
# This bar chart shows how each cluster differs across the features

selected_features = [
    "Reviewer_Score",
    "Positive_Lemma_Count",
    "Negative_Lemma_Count",
    "Num_Tags",
    "Total_Number_of_Reviews_Reviewer_Has_Given",
    "days_since_review"
]

# |%%--%%| <egyFAue58i|YIDIOVP49p>

assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
review_cluster_input = assembler.transform(cleaned)

# |%%--%%| <YIDIOVP49p|1Gz04Nx2g8>

print("Number of rows:", review_cluster_input.count())

# |%%--%%| <1Gz04Nx2g8|ePjkMJK7I0>

review_cluster_input.select(selected_features).printSchema()

# |%%--%%| <ePjkMJK7I0|lsooRzUIq2>

for col in selected_features:
    review_cluster_input.select(col).filter(isnan(col) | isnull(col)).show(1)

# |%%--%%| <lsooRzUIq2|NIiXvqxH41>

# Try with only one valid column (e.g., 'Average_Score' or 'Reviewer_Score') if it exists
cleaned.select("Reviewer_Score").filter("Reviewer_Score is not null").show()

# |%%--%%| <NIiXvqxH41|6PIgnADl4m>

kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=4, seed=42)
model = kmeans.fit(review_cluster_input)
review_clusters = model.transform(review_cluster_input)
