r"""°°°
# DSC511 Group Project: Hotel Review Sentiment Analysis and Rating Prediction

## Authors

- Maria Tsilidou
- Anastasios Nikodimou
- Ioannis Demetriou
°°°"""
# |%%--%%| <yBXH3hwc7S|GGUyxopChC>

# Importing libraries

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
import folium
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

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

- There are duplicate entries
- There are obvious erroneous entries
- There are missing features
°°°"""
# |%%--%%| <8zliPt5I2c|b9K4Vmutc1>

cleaned = original

# |%%--%%| <b9K4Vmutc1|jPAQaVYObc>

# Checking for duplicates
cleaned = cleaned.drop_duplicates()

print(f'From our dataset, {cleaned.count()} / {original.count()} are distinct')

# |%%--%%| <jPAQaVYObc|vO0cDou6S3>
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

#|%%--%%| <ydraWNJwES|ET3ONfqPae>
r"""°°°
We notice that some of our features are time related, but are typed as strings.

In particular,

- `Review_Date`
- `days_since_review`

We will cast them to datetime and integers (after we confirm the units) respectively
°°°"""
# |%%--%%| <ET3ONfqPae|qShVYDeqAO>

cleaned = cleaned\
    .withColumn('Review_Date', to_date(col('Review_Date'), format='M/d/yyyy'))

# Let's see if it worked
cleaned\
    .select('Review_Date')\
    .show(5)

#|%%--%%| <qShVYDeqAO|Fd4YrL2N5E>

cleaned\
    .select('days_since_review')\
    .show(5)

# We see that the format is "<days> day(s)"

def parse_days_ago(literal):
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


days_ago_udf = udf(parse_days_ago, IntegerType())

cleaned = cleaned\
    .withColumn('days_since_review',
        days_ago_udf(col('days_since_review'))
    )

# Admire our result
cleaned\
    .select('days_since_review')\
    .show(5)

# |%%--%%| <Fd4YrL2N5E|8bGrLv3gNR>

r"""°°°
## Geospatial

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
# |%%--%%| <8bGrLv3gNR|QSsRyOLHlW>
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

#|%%--%%| <QSsRyOLHlW|a1ApOgkdR7>

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

#|%%--%%| <a1ApOgkdR7|R3aUHprpeK>

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

#|%%--%%| <R3aUHprpeK|Gc9ZkdLZsv>
r"""°°°
We saw before that our dataset consists of the join between "Hotel", "Reviewer" and "Review".
It is natural to catagorize the "keys" of these tables where possible. In particular,
the natural categorizations are:

- `Reviewer_Nationality`
- `Hotel_Name`

Let's explore and encode them.
°°°"""
#|%%--%%| <Gc9ZkdLZsv|XVT2KTymC1>

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
#|%%--%%| <XVT2KTymC1|8TDxyIiSsX>
r"""°°°
We can see that from our dataset, the British, American and Australian tourists
seem to leave the most reviews. This might be because in our original dataset
all reviews are in English, and in these countries English is their native language.

It's worth noting that this selection was either enforced by the review website
or non-Engish reviews where filtered out before we received it, which appears
to introduce a bias toards more affluent countries.
°°°"""
#|%%--%%| <8TDxyIiSsX|dM3CKF2Yjk>

byHotel = cleaned\
    .groupBy('Hotel_Name')

print(f'There are reviews for {byHotel.count().count()} different hotels')
# TODO: Check if there are duplicates with similar name

byHotel\
    .count()\
    .sort('count', ascending=False)\
    .show(n=10, truncate=False)

hotel_indexer = StringIndexer(inputCol='Hotel_Name', outputCol='Hotel_Name_Encoded')

cleaned = hotel_indexer.fit(cleaned).transform(cleaned)
cleaned.select('Hotel_Name', 'Hotel_Name_Encoded').show(5)

#|%%--%%| <dM3CKF2Yjk|WZ7EuFtcim>

# At this point preprocessing has finished we save our data to parquet
cleaned.write.parquet('./data/Hotel_Reviews.parquet')

# |%%--%%| <WZ7EuFtcim|XnhwhjuVqD>
r"""°°°
### Explore Features
°°°"""
#|%%--%%| <XnhwhjuVqD|5R3d4gNbbg>
r"""°°°
Our dataset is quite large, and considering all observations when exploring the
dataset would be computational expensive and time consuming.

To overcome this, we sample from the original dataset.
°°°"""
# |%%--%%| <5R3d4gNbbg|LxF4mG1IId>

# Taking a smaller chunk to make exploration more computational efficient

sample = cleaned.sample(fraction=0.1, withReplacement=False, seed=42)

# |%%--%%| <LxF4mG1IId|hZiIKJbPyv>

# FIXME: For compatibility we cast to pandas - this should not be done in the content of big data

sample_pandas = sample.toPandas()
description = sample_pandas[['Additional_Number_of_Scoring', 'Average_Score', 'Reviewer_Score']].describe()

print('[Sample] Numerical features description:')
print(description)

# |%%--%%| <hZiIKJbPyv|Lw5aD9cErS>
r"""°°°
From the statistical summary above we can decide whether our data needs scaling
or not based on mean and std values.

Here mean ranges from 8.4 ('Average_Score') to 496.51 ('Additional_Number_of_Scoring').
This is an indication that scaling will be necessary since we are talking about a big
enough difference in the scale of our data.
°°°"""
# |%%--%%| <Lw5aD9cErS|izGJVPKzx0>
r"""°°°
#### Checking for skewness in the target variable
°°°"""
# |%%--%%| <izGJVPKzx0|1Wyw8wzBd8>

# Creating the histogram

plt.figure(figsize=(8, 4))  # Setting figure size
plt.hist(sample_pandas['Average_Score'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)

plt.title('Target Variable Distribution', fontsize=14, fontweight='bold', color='#2c3e50')
plt.xlabel('Average Score', fontsize=12, color='#34495e')
plt.ylabel('Frequency', fontsize=12, color='#34495e')

# Remove top and right spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show the plot
plt.show()

# |%%--%%| <1Wyw8wzBd8|bw3NG2DqU4>
r"""°°°
From the statistical summary we saw that the mean of the 'Average_Score' is close to 8.4
and from the plot we can see that our target variable is slightly skewed to the left.

This could potentially be due to the presence of an outlier around the value 6.3 - 6.4.
To be more specific about the skewness we can use .skew().
°°°"""
# |%%--%%| <bw3NG2DqU4|wuMxIcSJmL>

sample_pandas['Average_Score'].skew()

# |%%--%%| <wuMxIcSJmL|3FUDZWTAUR>
r"""°°°
#### Checking for linear correlation between numeric features and the target variable
°°°"""
# |%%--%%| <3FUDZWTAUR|BpO5D3ikW9>

# Setting Seaborn style for a cleaner look

sns.set_style("white")
sns.set_palette("coolwarm")

# Creating the pairplot
g = sns.pairplot(
    sample_pandas,
    x_vars=[
        'Additional_Number_of_Scoring', 'Review_Total_Negative_Word_Counts',
        'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts',
        'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score'
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

# |%%--%%| <BpO5D3ikW9|O0VfzYyfFv>
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

num_features = sample_pandas[[
        'Additional_Number_of_Scoring', 'Review_Total_Negative_Word_Counts',
        'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts',
        'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score', 'Average_Score'
    ]]
sns.heatmap(num_features.corr(method='pearson'), annot=True)

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

