Data Scientist's Roadmap
==========
*A minimalist roadmap to The Data Science World, based on [Kaggle's Roadmap](https://www.kaggle.com/learn/overview).*

# Python
*Learn the most important language for data science.*

## Hello, Python
*A quick introduction to Python syntax, variable assignment, and numbers. [#](https://www.kaggle.com/colinmorris/hello-python)*

### Variable Assignment

```python
=
```

### Function Calls

```python
func(var)
var.func()
```

### Numbers and Arithmetic in Python

```python
/     # true division
//    # floor division
%     # modulus
**    # exponentiation
```

### Built-in Functions for Working with Numbers

```python
min()
max()
abs()
# conversion functions
int()
float()
```

## Functions and Getting Help
*Calling functions and defining our own, and using Python's builtin documentation. [#](https://www.kaggle.com/colinmorris/functions-and-getting-help)*

### Getting Help
*on modules, objects, instances, and ...*

```python
help()
dir()
```

### Functions

```python
def func_name(vars):
    # some useful codes
    return # some useful results
```

#### Docstrings

```python
""" some useful info about the function """  # `help()` returns this
```

#### Functions w/o Return

```python
print()
```

#### Default Arguments

```python
print(..., sep='\t')
```

#### Functions Applied to Functions

```python
fn(fn(arg))
string.lower().split()
```

## Booleans and Conditionals
*Using booleans for branching logic. [#](https://www.kaggle.com/colinmorris/booleans-and-conditionals)*

### Booleans

```python
True
False
bool()
```

#### Comparison Operations

```python
a == b    # a equal to b
a != b    # a not equal to b
a <  b    # a less than b
a >  b    # a greater than b
a <= b    # a less than or equal to b
a >= b    # a greater than or equal to b
```

#### Order of Operators
*PEMDAS combined with Boolean Values*

```python
()
**
+x, -x, ~x
*, /, //, %
+, -
<<, >>
&
^
|
==, !=, >, >=, <, <=, is, is not, in, not in
not
and
or
```

### Conditionals

```python
if
elif
else
```

#### Trues and Falses

- All numbers are treated as `True`, except `0`.
- All strings are treated as `True`, except the empty string `""`.
- Empty sequences (strings, lists, tuples, sets)  are `False` and the rest are `True`.

#### Conditional Expressions
*Setting a variable to either of two values depending on a condition.*

```python
outcome = 'failed' if grade < 50 else 'passed'
```

## Lists
*Lists and the things you can do with them. Includes indexing, slicing and mutating. [#](https://www.kaggle.com/colinmorris/lists)*

### Lists
*A mutable mix of same or different types of variables*

```python
[]
list()
```

#### Indexing

```python
planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]

# first element
planets[0]
# last element
planets[-1]
```

#### Slicing

```python
planets[:3]
planets[-3:]
```

#### Changing Lists

```python
planets[:3] = ['Mur', 'Vee', 'Ur']
```

#### List Functions

```python
len()
sorted()
max()
sum()
any()
```

#### Python Attributes & Methods
*Everything is an Object.*

```python
# complex number object
c = 12 + 5j
c.imag
c.real
```

```python
# integer number object
x = 12
x.bit_length()
```

#### List Methods

```python
list.append()
list.pop()
list.index()
in
```

### Tuples
*Immutable.*

```python
()
,
tuple()
```

```python
x = 0.125
numerator, denominator = x.as_integer_ratio()
```

## Loops and List Comprehensions
*For and while loops, and a much-loved Python feature: list comprehensions. [#](https://www.kaggle.com/colinmorris/loops-and-list-comprehensions)*

### Loops
*Use in every iteratable objects: list, tuples, strings, ...*

```python
for - in - :
    # some useful codes
```

- `range()`
- `while`

### List Comprehensions

```python
[- for - in -]
```

```python
squares = [n**2 for n in range(10)]
# constant
[32 for planet in planets]
```

```python
# with if
short_planets = [planet.upper() + "!" for planet in planets if len(planet) < 6]
```

```python
# combined with other functions
return len([num for num in nums if num < 0])
return sum([num < 0 for num in nums])
return any([num % 7 == 0 for num in nums])
```

Solving a problem with less code is always nice, but it's worth keeping in mind the following lines from **The Zen of Python**.
> Readability counts.<br>
> Explicit is better than implicit.

### Enumerate

```python
for index, item in enumerate(items):
    # some useful codes
```

## String and Directories
*Working with strings and dictionaries, two fundamental Python data types. [#](https://www.kaggle.com/colinmorris/strings-and-dictionaries)*

### Strings
*Immutable.*

```python
''
""
""" """
str()
```

```python
[char + '! ' for char in "Planet"]
>>> ['P! ', 'l! ', 'a! ', 'n! ', 'e! ', 't! ']
```

```python
"Planet"[0] = 'M'
>>> TypeError: 'str' object does not support item assignment
```

#### String Methods

```python
str.upper()
str.lower()
str.index()
str.startswith()
str.endswith()
```

#### String and List, Back and Forward

```python
# split
year, month, day = '2020-03-05'.split('-')
year, month, day
>>> ('2020', '03', '05')
```

```python
# join
'/'.join([month, day, year])
>>> '03/05/2020'
```

#### String Formatting

```python
"{}".format()
f"{}"
```

### Dictionaries
*Pairs of keys,values.*

```python
{}
dict()
```

```python
numbers = {'one':1, 'two':2, 'three':3}
numbers['one']
numbers['eleven'] = 11
```

#### Dictionary Comprehensions

```python
planet_to_initial = {planet: planet[0] for planet in planets}
```

#### Access to all Keys or all Values

```python
dict.keys()
dict.values()
```

```python
' '.join(sorted(planet_to_initial.values()))
```

#### Get key by value

```python
key_of_min_value = min(numbers, key=numbers.get)
```

#### `in`

```python
'M' in planet_to_initial.values()
>>> True
```

#### Loops in Dictionaries

```python
# loop over keys
for planet in planet_to_initial:
    print(planet)
```

```python
# loop over (key, value) pairs using `item`
for planet, initial in planet_to_initial.items():
    print(f"{planet} begins with \"{initial}\"")
```

## Working with External Libraries
*Imports, operator overloading, and survival tips for venturing into the world of external libraries. [#](https://www.kaggle.com/colinmorris/working-with-external-libraries)*

### Imports

```python
# simple import, `.` access
import math
math.pi
```

```python
# `as` import, short `.` access
import math as mt
mt.pi
```

```python
# `*` import, simple access
from math import *
pi
```

The problem of `*` import is that some modules (ex. `math` and `numpy`) have functions with same name (ex. `log`) but with different semantics. So one of them overwrites (or "shadows") the other. It is called **overloading**.

```python
# combined, solution for the `*` import
from math import log, pi
from numpy import asarray
```

### Submodules

Modules contain variables which can refer to functions or values. Sometimes they can also have variables referring to other modules.

```python
import numpy
dir(numpy.random)
>>> ['set_state', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'test', 'triangular', 'uniform', ...]
```

```python
# make an array of random numbers
rolls = numpy.random.randint(low=1, high=6, size=10)
rolls
>>> array([3, 2, 5, 2, 4, 2, 2, 3, 2, 3])
```

#### Get Help

Standard Python datatypes are: `int`, `float`, `bool`, `list`, `str`, and `dict`.

As you work with various libraries for specialized tasks, you'll find that they define their own types. For example

- Matplotlib: `Subplot`, `Figure`, `TickMark`, and `Annotation`
- Pandas: `DataFrame` and `Series`
- Tensorflow: `Tensor`

Use `type()` to find the type of an object. Use `dir()` and `help()` for more details.

```python
dir(umpy.ndarray)
>>> [...,'__bool__', ..., '__delattr__', '__delitem__', '__dir__', ..., '__sizeof__', ..., 'max', 'mean', 'min', ..., 'sort', ..., 'sum', ..., 'tobytes', 'tofile', 'tolist', 'tostring', ...]
```

### Operator Overloading

#### Index

```python
# list
xlist = [[1,2,3], [2,4,6]]
xlist[1,-1]
>>> TypeError: list indices must be integers or slices, not tuple
```

```python
# numpy array
xarray = numpy.asarray(xlist)
xarray[1,-1]
>>> 6
```

#### Add

```python
# list
[3, 4, 1, 2, 2, 1] + 10
>>> TypeError: can only concatenate list (not "int") to list
```

```python
# numpy array
rolls + 10
>>> array([13, 12, 15, 12, 14, 12, 12, 13, 12, 13])
```

```python
# tensorflow
import tensorflow as tf
a = tf.constant(1)
b = tf.constant(1)
a + b
>>> <tf.Tensor 'add:0' shape=() dtype=int32>
```

When Python programmers want to define how operators behave on their types, they do so by implementing **Dunder (Special) Methods**, methods with special names beginning and ending with 2 underscores such as `__add__` or `__contains__`. More info: https://is.gd/3zuhhL

# Pandas
*Solve short hands-on challenges to perfect your data manipulation skills.*

## Creating, Reading and Writing
*You can't work with data if you can't read it. [#](https://www.kaggle.com/residentmario/creating-reading-and-writing)*

### Creating Data

#### DataFrame

It is a **table** and contains an array of individual entries. Each entry corresponds to a row (or record) and a column.

```python
import pandas as pd
pd.DataFrame({"Apples": [50, 21], "Bananas": [131, 2]})
```

The syntax for declaring a new one is a **dictionary** whose keys are the column names, and whose values are a list of entries.

The list of row labels used in a DataFrame is known as an Index. We can assign values to it by using an `index` parameter in our constructor.

```python
pd.DataFrame({"Apples": [50, 21], "Bananas": [131, 2]}, index=["2018 Sales", "2019 Sales"])
```

#### Series

In essence, it is a single **column** of a DataFrame, a sequence of data values.

```python
pd.Series([30, 50, 21])
```

You can assign column values to the Series the same way as before, using an `index` parameter. However, a Series does not have a column name, it only has one overall `name`.

```python
pd.Series([30, 50, 21], index=["2017 Sales", "2018 Sales", "2019 Sales"], name="Apples")
```

### Reading Data Files

Data can be stored in any of a number of different forms and formats. By far the most basic of these is the humble CSV file. A CSV file is a table of values separated by commas.

```python
# load data
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
```

- If your CSV file has a built-in index, pandas can use that column for the index (instead of creating a new one automatically).

```python
# data dimention
wine_reviews.shape

# columns' name
wine_reviews.columns

# top rows
wine_reviews.head()

# bottom rows
wine_reviews.tail()
```

### Writing Data to File

```python
animals = pd.DataFrame({"Cows": [12, 20], "Goats": [22, 19]}, index=["Year 1", "Year 2"])
animals.to_csv("cows_and_goats.csv")
```

## Indexing, Selecting & Assigning
*Pro data scientists do this dozens of times a day. You can, too! [#](https://www.kaggle.com/residentmario/indexing-selecting-assigning)*

### Naive Accessors

In Python, we can access the property of an object by accessing it as an attribute. A `reviews` object, might have a `country` property, which we can access by calling `reviews.country`. Columns in a pandas DataFrame work in much the same way.

If we have a Python dictionary, we can access its values using the indexing `[]` operator.

```python
# select the `country` column
reviews["country"]
```

A pandas Series looks kind of like a dictionary. So, to drill down to a single specific value, we need only use the indexing operator `[]` once more.

```python
# select the first value from the `country` column
reviews["country"][0]
>>> 'Italy'
```

### Indexing in Pandas

For more advanced operations, pandas has its own accessor operators, `iloc` and `loc`.

#### Index-based Selection
*Selecting data based on its **numerical position** in the data, like a matrix.*

```python
# select the first row
reviews.iloc[0]

# select the first column, `:` means everything
reviews.iloc[:, 0]

# select the first value from the `country` column
reviews["country"].iloc[0]

# select the last five elements of the dataset
reviews.iloc[-5:]
```

#### Label-based Selection
*Selecting data based on its **index value**, with **inclusive** range.*

```python
# select the first value from the `country` column
reviews.loc[0, "country"]

# select all the entries from three specific columns
reviews.loc[:, ["taster_name", "taster_twitter_handle", "points"]]
```

#### Inclusive Range, `iloc` vs. `loc`

```python
# select first three rows
reviews.iloc[:3]
# or
reviews.loc[:2]
```

```python
# select the first 100 records of the `country` and `variety` columns.
cols_idx = [0, 11]
reviews.iloc[:100, cols_idx]
# or
cols = ["country", "variety"]
reviews.loc[:99, cols]
```

#### Manipulating the Index

```python
reviews.set_index("title")
```

### Conditional Selection

To do interesting things with the data, we often need to ask questions based on conditions.

To combine multiple conditions in pandas, **bitwise operators** must be used.

```python
&    # AND          x & y
|    # OR           x | y
^    # XOR          x ^ y
~    # NOT          ~x
>>   # right shift  x>>
<<   # left shift   x<<
```

For example, suppose that we're interested in better-than-average wines produced in Italy.

```python
cond1 = (reviews["country"] == "Italy")
cond2 = (reviews["points"] >= 90)
reviews.loc[cond1 & cond2]
```

#### Built-in Conditional Selectors

`isin()` lets you select data whose value "is in" a list of values.

```python
# select wines only from Italy or France
reviews.loc[reviews["country"].isin(["Italy", "France"])]
```

`isna()` and `notna()` let you highlight values which are (or are not) empty (NaN).

```python
# filter out wines lacking a price tag in the dataset
reviews.loc[reviews["price"].notna()]
```

- `isnull()` is an alias for `isna()`, same as `notnull()` for `notna()`.

### Assigning Data

```python
# you can assign either a constant value
reviews["critic"] = "everyone"

# or with an iterable of values
reviews["index_backwards"] = range(len(reviews), 0, -1)
```

## Summary Functions and Maps
*Extract insights from your data. [#](https://www.kaggle.com/residentmario/summary-functions-and-maps)*

### Summary Functions

```python
# get summary statistic about a DataFrame
reviews.describe()
```

- `count`: shows how many rows have non-missing values.
- `mean`: the average.
- `std`: the standard deviation, measures how numerically spread out the values are.
- `min`, `25%` (25th percentile), `50%` (50th percentiles), `75%` (75th percentiles) and `max`

```python
# get summary statistic about a Series
reviews["points"].describe()

# see the mean
reviews["points"].mean()

# see a list of unique values
reviews["points"].unique()

# see a list of unique values and how often they occur
reviews["points"].value_counts()

# get the titles & points of the 3 highest point
reviews["points"].nlargest(3)
```

```python
# get the title of the wine with the highest points-to-price ratio
max_p2pr = (reviews["points"] / reviews["price"]).idxmax()
reviews.loc[max_p2pr, "title"]
>>> 'Bandit NV Merlot (California)'
```

### Maps

A function that takes one set of values and "maps" them to another set of values, for creating new representations from existing data.

#### `map()`

```python
# remean the scores the wines received to 0
review_points_mean = reviews["points"].mean()
reviews["points"].map(lambda p: p - review_points_mean)
```

```python
# create a series counting how many times each of "tropical" or "fruity" appears in the description column
n_tropical = reviews["description"].map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews["description"].map(lambda desc: "fruity" in desc).sum()
pd.Series([n_tropical, n_fruity], index=["tropical", "fruity"])
```

The function you pass to `map()` should expect a single value from the Series (a point value, in the above example), and return a transformed version of that value. `map()` returns a new **Series**.

#### `apply()`

```python
# remean the scores the wines received to 0
def remean_points(row):
    row["points"] = row["points"] - review_points_mean
    return row
reviews.apply(remean_points, axis="columns")
```

```python
# create a series with the number of stars corresponding to each review
def stars(row):
    # any wines from country X should automatically get 3 stars, because of ADs' MONEY!
    if row["country"] == "X":
        return 3
    elif row["points"] >= 95:
        return 3
    elif row["points"] >= 85:
        return 2
    else:
        return 1
reviews.apply(stars, axis="columns")
```

`apply()` is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row. `apply()` returns a new **DataFrame**.

If we had called `reviews.apply()` with `axis="index"`, then instead of passing a function to transform each row, we would need to give a function to transform each column.

#### Pandas built-ins Common Mapping Operators

They perform a simple operation between a lot of values on the left and a single (a lot of) value(s) on the right.

```python
# remean the scores the wines received to 0
review_points_mean = reviews["points"].mean()
reviews["points"] - review_points_mean
```

```python
# combine country and region information in the dataset
reviews["country"] + " - " + reviews["region_1"]
```

These operators are **faster** than `map()` or `apply()` because they uses speed ups built into pandas. All of the standard Python operators (`>`, `<`, `==`, and so on) work in this manner.

However, they are **not as flexible as** `map()` or `apply()`, which can do more advanced things, like applying conditional logic, which cannot be done with addition and subtraction alone.

## Grouping and Sorting
*Scale up your level of insight. The more complex the dataset, the more this matters. [#](https://www.kaggle.com/residentmario/grouping-and-sorting)*

### Groupwise Analysis

Maps allow us to transform data in a DataFrame or Series one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in.

#### `groupby()`

```python
# replicate what `value_counts()` does
reviews.groupby("points")["points"].count()

# get the minimum price from each group of points
reviews.groupby("points")["price"].min()

# get a series whose index is the taster_twitter_handle values count how many reviews each person wrote
reviews.groupby("taster_twitter_handle").size()
# or
reviews.groupby("taster_twitter_handle")["taster_twitter_handle"].count()

# get the title of the first wine reviewed from each winery
reviews.groupby("winery").apply(lambda df: df["title"].iloc[0])
```

#### Aggregate Different Functions Simultaneously

```python
# get a dataframe whose index is the variety category and values are the `min` and `max` prices
reviews.groupby("variety")["price"].agg([min, max])
```

#### Multi-indexes, Group by More than One Column

```python
# pick out the best wine by country and province
reviews.groupby(["country", "province"]).apply(lambda df: df.loc[df["points"].idxmax()])
```

Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices.

They also require two levels of labels to retrieve a value.

The use cases for a multi-index are detailed alongside instructions on using them in the [MultiIndex / Advanced Selection](https://pandas.pydata.org/pandas-docs/stable/advanced.html) section of the pandas documentation.

```python
# convert back to a regular index
count_prov_best.reset_index()
```

### Sorting

#### `sort_values()`

```python
# sort (country, province) based on how many reviews are belong to
count_prov_reviewed = reviews.groupby(["country", "province"])["description"].agg([len])
count_prov_reviewed.reset_index().sort_values(by="len", ascending=False)
```

#### Sort by More than One Column

```python
count_prov_reviewed.reset_index().sort_values(by=["country", "len"], ascending=False)
```

#### `sort_index()`

```python
# get a series whose index is wine prices and values is the maximum points a wine costing that much was given in a review. sort the values by price, ascending
reviews.groupby("price")["points"].max().sort_index(ascending=True)
```

## Data Types and Missing Values
*Deal with the most common progress-blocking problems. [#](https://www.kaggle.com/residentmario/data-types-and-missing-values)*

### Dtypes

The data type for a column in a DataFrame or a Series is known as the `dtype`.
  - `int64`, `float64`, `object`

```python
# a dataframe
reviews.dtypes

# a series
reviews["price"].dtype

# a dataframe or series index
reviews.index.dtype

# convert a dtype
reviews["points"].astype("float64")
```

### Missing Values (NaNs)

#### `isna()`, `notna()`

```python
# get a series of True & False, based on where NaNs are
reviews["price"].isna()

# find the number of NaNs
reviews["price"].isna().sum()

# create a dataframe of rows with missing country
reviews[reviews["country"].isna()]
```

- `isnull()` is an alias for `isna()`, same as `notnull()` for `notna()`.

#### `fillna()`

```python
# fill NaNs with Unknown
reviews["region_1"].fillna("Unknown")
```

#### `replace()`

```python
# replace missing data which is given some kind of sentinel values
reviews["region_1"].replace(["Unknown", "Undisclosed", "Invalid"], "NaN")
```

#### `dropna()`

```python
# filter rows with NaNs
reviews.dropna(axis=0)

# filter columns with NaNs
reviews.dropna(axis=1)
```

## Renaming and Combining
*Data comes in from many sources. Help it all make sense together. [#](https://www.kaggle.com/residentmario/renaming-and-combining)*

### Renaming

#### `rename()`

```python
# change the names of columns
reviews.rename(columns={"region_1": "region", "region_2": "locale"})

# change the indices of rows
reviews.rename(index={0: "firstEntry", 1: "secondEntry"})
```

### `raname_axis()`

```python
# change the names of axes, form rows to wines, from columns to fields
reviews.rename_axis("wines", axis="rows").rename_axis("fields", axis="columns")
```

### Combining

We will sometimes need to combine different DataFrames and/or Series. Pandas has three core methods for doing this. In order of increasing complexity, these are

#### `concat()`

It will smush a list of elements together along an axis.

This is useful when we have data in different DataFrame or Series objects but having the same columns.

```python
canadian_yt = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_yt = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_yt, british_yt])
```

#### `join()`

It lets you combine different DataFrame objects which have an index in common.

```python
# pull down videos that happened to be trending on the same day in both Canada and the UK
left = canadian_yt.set_index(["title", "trending_date"])
right = british_yt.set_index(["title", "trending_date"])
left.join(right, lsuffix="_CAN", rsuffix="_UK")
```

- The `lsuffix` and `rsuffix` parameters are necessary when the data has the same column names in both datasets.

#### `merge()`

# Data Visualization
*Make great data visualizations. A great way to see the power of coding!*

## Line Charts
*Visualize trends over time. [#](https://www.kaggle.com/alexisbcook/line-charts)*

### Set up the notebook

```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

### Line Chart

```python
# load a timeseries data file
spotify_data = pd.read_csv("../input/spotify.csv", index_col="Date", parse_dates=True)

# set the width and height of the figure
plt.figure(figsize=(14,6))

# add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# plot a line chart for daily global streams of each song
sns.lineplot(data=spotify_data)

# plot a subset of the data
sns.lineplot(data=spotify_data["Shape of You"], label="Shape of You")

# add label for horizontal axis
plt.xlabel("Date")
```

## Bar Charts and Heatmaps
*Use color or length to compare categories in a dataset. [#](https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps)*

### Bar Chart

```python
# load data
flight_data = pd.read_csv("../input/flight_delays.csv", index_col="Month")

# add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# rotate labels for horizontal axis
plt.xticks(rotation="vertical")

# plot a bar chart, showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data["NK"])

# add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")
```

- **Note**: You must select the indexing column with `flight_data.index`, and it is not possible to use `flight_data['Month']`, because when we loaded the dataset, the `"Month"` column was used to index the rows.

### Heatmap

```python
# add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# plot a heatmap, showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)

# add label for horizontal axis
plt.xlabel("Airline")
```

```python
# get the maximum average delay on March
flight_data.loc[3].max()

# find the aireline with the minimum average delay on October
flight_data.loc[10].idxmin()
```

## Scatter Plots
*Leverage the coordinate plane to explore relationships between variables. [#](https://www.kaggle.com/alexisbcook/scatter-plots)*

### Scatter Plots

```python
# load data
insurance_data = pd.read_csv("../input/insurance.csv")

# a simple scatter plot
sns.scatterplot(x=insurance_data["bmi"], y=insurance_data["charges"])

# add a regression line
sns.regplot(x=insurance_data["bmi"], y=insurance_data["charges"])

# a color-coded scatter plot
sns.scatterplot(x=insurance_data["bmi"], y=insurance_data["charges"], hue=insurance_data["smoker"])

# add two regression lines, corresponding to hue
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

# a categorical scatter plot with non-overlapping points (swarmplot)
sns.swarmplot(x=insurance_data["smoker"], y=insurance_data["charges"])
```

## Distributions
*Create histograms and density plots. [#](https://www.kaggle.com/alexisbcook/distributions)*

### Histogram

```python
# load data
iris_data = pd.read_csv("../input/iris.csv", index_col="Id")

# a simple histogram
sns.distplot(a=iris_data["Petal Length (cm)"], kde=False)

# a kde (kernel density estimate) plot
sns.kdeplot(data=iris_data["Petal Length (cm)"], shade=True)

# a 2D kde plot
sns.jointplot(x=iris_data["Petal Length (cm)"], y=iris_data["Sepal Width (cm)"], kind="kde")
```

### Color-coded Plots

```python
# load data
iris_set_data = pd.read_csv("../input/iris_setosa.csv", index_col="Id")
iris_ver_data = pd.read_csv("../input/iris_versicolor.csv", index_col="Id")
iris_vir_data = pd.read_csv("../input/iris_virginica.csv", index_col="Id")

# kde plots for each one, histograms can be used too
sns.kdeplot(data=iris_set_data["Petal Length (cm)"], label="Setosa", shade=True)
sns.kdeplot(data=iris_ver_data["Petal Length (cm)"], label="Versicolor", shade=True)
sns.kdeplot(data=iris_vir_data["Petal Length (cm)"], label="Virginica", shade=True)

# force legend to appear
plt.legend()
```

## Choosing Plot Types
*Decide how to best tell the story behind your data. [#](https://www.kaggle.com/alexisbcook/choosing-plot-types-and-custom-styles)*

### Trends

- A trend is defined as a pattern of **change**.
- `sns.lineplot` - Line charts are best to show trends over a period of time, and multiple lines can be used to show trends in more than one group.

### Relationship

- `sns.barplot` - Bar charts are useful for comparing quantities corresponding to different groups.
- `sns.heatmap` - Heatmaps can be used to find color-coded patterns in tables of numbers.
- `sns.scatterplot` - Scatter plots show the relationship between two continuous variables; if color-coded, we can also show the relationship with a third categorical variable.
- `sns.regplot` - Including a regression line in the scatter plot makes it easier to see any linear relationship between two variables.
- `sns.lmplot` - This command is useful for drawing multiple regression lines, if the scatter plot contains multiple, color-coded groups.
- `sns.swarmplot` - Categorical scatter plots show the relationship between a continuous variable and a categorical variable.

### Distribution

- A distribution shows the possible values that we can **expect** to see in a variable, along with how likely they are.
- `sns.distplot` - Histograms show the distribution of a single numerical variable.
- `sns.kdeplot` - KDE plots (or 2D KDE plots) show an estimated, smooth distribution of a single numerical variable (or two numerical variables).
- `sns.jointplot` - This command is useful for simultaneously displaying a 2D KDE plot with the corresponding KDE plots for each individual variable.

## Final Project
*Practice for real-world application. [#](https://www.kaggle.com/alexisbcook/final-project)*

### Use your own dataset

```python
# list all your datasets' folders
import os
print(os.listdir("../input"))
```

## Creating Your Own Notebooks
*How to put your new skills to use for your next personal or work project. [#](https://www.kaggle.com/alexisbcook/creating-your-own-notebooks)*

# Geospatial Analysis
*Create interactive maps, and discover patterns in geospatial data.*

## Your First Map
*Get started with plotting in GeoPandas. [#](https://www.kaggle.com/alexisbcook/your-first-map)*

### Introduction

With this course you can find solutions for several real-world problems like:

- Where should a global non-profit expand its reach in remote areas of the Philippines?
- How do purple martins, a threatened bird species, travel between North and South America? Are the birds travelling to conservation areas?
- Which areas of Japan could potentially benefit from extra earthquake reinforcement?
- Which Starbucks stores in California are strong candidates for the next Starbucks Reserve Roastery location?
- ...

### Reading Data

```python
import geopandas as gpd
```

The data was loaded into a (GeoPandas) `GeoDataFrame` object has all of the capabilities of a (Pandas) DataFrame. So, every command that you can use with a DataFrame will work with the data!

There are many, many different geospatial file formats, such as [shapefile](https://en.wikipedia.org/wiki/Shapefile), [GeoJSON](https://en.wikipedia.org/wiki/GeoJSON), [KML](https://en.wikipedia.org/wiki/Keyhole_Markup_Language), and [GPKG](https://en.wikipedia.org/wiki/GeoPackage).

- shapefile is the most common file type that you'll encounter, and
- all of these file types can be quickly loaded with the `read_file()` function.

Every GeoDataFrame contains a special `geometry` column. It contains all of the geometric objects that are displayed when we call the `plot()` method. While this column can contain a variety of different datatypes, each entry will typically be a `Point`, `LineString`, or `Polygon`.

### Create Your Map
*Create it layer by layer.*

```python
# load data
world_loans = gpd.read_file(
    "../input/geospatial-learn-course-data/kiva_loans/kiva_loans/kiva_loans.shp"
)

# define a base map with county boundaries
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
ax = world.plot(figsize=(20, 20), color="whitesmoke", linestyle=":", edgecolor="lightgray")

# add loans to the base map
world_loans.plot(ax=ax, color="black", markersize=2)
```

You can subset the data for more details.

```python
# subset the data
phl_loans = world_loans.loc[world_loans["country"] == "Philippines"].copy()

# enable fiona driver & load a KML file containing island boundaries
gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
phl = gpd.read_file("../input/geospatial-learn-course-data/Philippines_AL258.kml", driver="KML")

# define a base map with county boundaries
ax_ph = phl.plot(figsize=(20, 20), color="whitesmoke", linestyle=":", edgecolor="lightgray")

# add loans to the base map
phl_loans.plot(ax=ax_ph, color="black", markersize=2)
```

## Coordinate Reference Systems
*It's pretty amazing that we can represent the Earth's surface in 2 dimensions! [#](https://www.kaggle.com/alexisbcook/coordinate-reference-systems)*

### Introduction

The world is a three-dimensional globe. So we have to use a map projection method to render it as a flat surface. Map projections can't be 100% accurate. Each projection distorts the surface of the Earth in some way, while retaining some useful property.

- The equal-area projections preserve **area**.
- The equidistant projections preserve **distance**.

We use a coordinate reference system (CRS) to show how the projected points correspond to real locations on Earth. CRSs are referenced by [European Petroleum Survey Group (EPSG)](http://www.epsg.org/) codes.

### Setting the CRS

When we create a GeoDataFrame from a shapefile, the CRS is already imported for us. But when creating a GeoDataFrame from a CSV file, we have to set the CRS to [EPSG 4326](https://epsg.io/4326), corresponds to coordinates in latitude and longitude.

```python
# create a DataFrame with health facilities in Ghana
import pandas as pd
facilities_df = pd.read_csv("../input/geospatial-learn-course-data/ghana/ghana/health_facilities.csv")

# convert the DataFrame to a GeoDataFrame
import geopandas as gpd
facilities = gpd.GeoDataFrame(facilities_df, geometry=gpd.points_from_xy(facilities_df.Longitude, facilities_df.Latitude))

# set the CRS code
facilities.crs = {"init": "epsg:4326"}
```

- We begin by creating a DataFrame containing columns with latitude and longitude coordinates.
- To convert it to a GeoDataFrame, we use `gpd.GeoDataFrame()`.
- The `gpd.points_from_xy()` function creates Point objects from the latitude and longitude columns.

### Re-projecting

Re-projecting refers to the process of changing the CRS. This is done in GeoPandas with the `to_crs()` method. For example, when plotting multiple GeoDataFrames, it's important that they all use the same CRS.

```python
# load a GeoDataFrame containing regions in Ghana
regions = gpd.read_file(
    "../input/geospatial-learn-course-data/ghana/ghana/Regions/Map_of_Regions_in_Ghana.shp"
)
regions.crs
>>> 32630
```

```python
# create a map
ax = regions.plot(figsize=(8, 8), color="whitesmoke", linestyle=":", edgecolor="black")
facilities.to_crs(epsg=32630).plot(ax=ax, alpha=0.6, markersize=1, zorder=1)
```

In case the EPSG code is not available in GeoPandas, we can change the CRS with what's known as the "proj4 string" of the CRS. The proj4 string to convert to latitude/longitude coordinates is:

```python
# change the CRS to EPSG 4326
regions.to_crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
```

### Attributes of Geometric Objects

For an arbitrary GeoDataFrame, the type in the `geometry` column depends on what we are trying to show: for instance, we might use:

- a `Point` for the epicenter of an earthquake,
- a `LineString` for a street, or
- a `Polygon` to show country boundaries.

All three types of geometric objects have built-in attributes that you can use to quickly analyze the dataset.

```python
# get the x- or y-coordinates of a point from the x and y attributes
facilities["geometry"].x

# calculate the area (in square kilometers) of all polygons
sum(regions["geometry"].to_crs(epsg=3035).area) / 10**6
```

-  [ESPG 3035](https://epsg.io/3035) Scope: Statistical mapping at all scales and other purposes where **true area** representation is required.

### Techniques from the Exercise

```python
# load data
import pandas as pd
birds_df = pd.read_csv(
    "../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=["timestamp"]
)

# create the GeoDataFrame
import geopandas as gpd
birds = gpd.GeoDataFrame(
    birds_df, geometry=gpd.points_from_xy(birds_df["location-long"], birds_df["location-lat"])
)

# create GeoDataFrame showing path for each bird
from shapely.geometry import LineString
path_df = (
    birds.groupby("tag-local-identifier")["geometry"]
    .apply(list)
    .apply(lambda x: LineString(x))
    .reset_index()
)
path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df["geometry"])
path_gdf.crs = {"init": "epsg:4326"}
```

## Interactive Maps
*Learn how to make interactive heatmaps, choropleth maps, and more! [#](https://www.kaggle.com/alexisbcook/interactive-maps)*

### The Data

```python
# load data
import pandas as pd
crimes = pd.read_csv(
    "../input/geospatial-learn-course-data/crimes-in-boston/crimes-in-boston/crime.csv",
    encoding="latin-1",
)

# drop rows with missing locations
crimes.dropna(subset=["Lat", "Long", "DISTRICT"], inplace=True)

# focus on major crimes in 2018
crimes = crimes[
    crimes["OFFENSE_CODE_GROUP"].isin(
        [
            "Larceny",
            "Auto Theft",
            "Robbery",
            "Larceny From Motor Vehicle",
            "Residential Burglary",
            "Simple Assault",
            "Harassment",
            "Ballistics",
            "Aggravated Assault",
            "Other Burglary",
            "Arson",
            "Commercial Burglary",
            "HOME INVASION",
            "Homicide",
            "Criminal Harassment",
            "Manslaughter",
        ]
    )
]
crimes = crimes[crimes["YEAR"] == 2018]

# focus on daytime robberies
daytime_robberies = crimes[
    ((crimes["OFFENSE_CODE_GROUP"] == "Robbery") & (crimes["HOUR"].isin(range(9, 18))))
]
```

### Base Map

In this tutorial, you'll learn how to create interactive maps with the `folium` package. We create the base map with `folium.Map()`.

```python
# create the base map
from folium import Map
base_map = Map(location=[42.32, -71.0589], tiles="openstreetmap", zoom_start=10)
```

- `location` sets the initial center of the map. We use the latitude (42.32° N) and longitude (-71.0589° E) of the city of Boston.
- `tiles` changes the styling of the map; in this case, we choose the OpenStreetMap style. If you're curious, you can find the other options listed [here](https://github.com/python-visualization/folium/tree/master/folium/templates/tiles).
- `zoom_start` sets the initial level of zoom of the map, where higher values zoom in closer to the map.

### Markers

We add markers to the map with `folium.Marker()`. Each marker below corresponds to a different robbery.

```python
# define the base map
map_marker = map_base

# add points to the map
from folium import Marker
for idx, row in daytime_robberies.iterrows():
    Marker([row["Lat"], row["Long"]], popup=row["HOUR"]).add_to(map_marker)

# display the map
map_marker
```

### Markers' Cluster

If we have a lot of markers to add, `folium.plugins.MarkerCluster()` can help to declutter the map. Each marker is added to a `MarkerCluster` object.

```python
# define the base map
map_cluser = map_base

# add points to the map
from folium.plugins import MarkerCluster
mc = MarkerCluster()
from folium import Marker
import math
for idx, row in daytime_robberies.iterrows():
    if not math.isnan(row["Lat"]) and not math.isnan(row["Long"]):
        mc.add_child(Marker([row["Lat"], row["Long"]]))

map_cluser.add_child(mc)

# display the map
map_cluser
```

- We used `math.isnan()` because `row["Lat"]` or `row["Long"]` are `float`.

### Bubble Maps

A bubble map uses circles instead of markers. By varying the size and color of each circle, we can also show the relationship between location and two other variables.

We create a bubble map by using `folium.Circle()` to iteratively add circles.

```python
# define the base map
map_bubble = map_base

# define color/size producer function
def color_producer(val):
    if val <= 12:
        # robberies that occurred in hours 9-12
        return "forestgreen"
    else:
        # robberies from hours 13-17
        return "darkred"

# add a bubble map to the base map
from folium import Circle
for i in range(len(daytime_robberies)):
    Circle(
        location=[daytime_robberies.iloc[i]["Lat"], daytime_robberies.iloc[i]["Long"]],
        radius=20,
        color=color_producer(daytime_robberies.iloc[i]["HOUR"]),
    ).add_to(map_bubble)

# display the map
map_bubble
```

- `location` is a list containing the center of the circle, in latitude and longitude.
- `radius` sets the radius of the circle.
  - We can implement this by defining a function similar to the `color_producer()` function that is used to vary the color of each circle.
- `color` sets the color of each circle.
  - `The color_producer()` function is used to visualize the effect of the hour on robbery location.


### Heatmaps

To create a heatmap, we use `folium.plugins.HeatMap()`. This shows the density of crime in different areas of the city, where red areas have relatively more criminal incidents.

```python
# define the base map
map_heat = map_base

# add a heatmap to the base map
from folium.plugins import HeatMap
HeatMap(data=crimes[["Lat", "Long"]], radius=10).add_to(map_heat)

# display the map
map_heat
```

- `data` is a DataFrame containing the locations that we'd like to plot.
- `radius` controls the smoothness of the heatmap. Higher values make the heatmap look smoother.

### Choropleth Maps

To understand how crime varies by police district, we'll create a choropleth map. To create a choropleth, we use `folium.Choropleth()`.

As a first step, we create a GeoDataFrame where each district is assigned a different row, and the `geometry` column contains the geographical boundaries.

```python
# create GeoDataFrame with geographical boundaries of districts
import geopandas as gpd
districts_full = gpd.read_file(
    "../input/geospatial-learn-course-data/Police_Districts/Police_Districts/Police_Districts.shp"
)
districts = districts_full[["DISTRICT", "geometry"]].set_index("DISTRICT")
```

```python
# create a Pandas Series shows the number of crimes in each police district
plot_dict = crimes["DISTRICT"].value_counts()
```

- It's very important that `plot_dict` has the same index as districts - this is how the code knows how to match the geographical boundaries with appropriate colors.

```python
# define the base map
map_choropleth = map_base

# add a choropleth map to the base map
from folium import Choropleth
Choropleth(
    geo_data=districts.__geo_interface__,
    data=plot_dict,
    key_on="feature.id",
    fill_color="YlGnBu",
    legend_name="Major Criminal Incidents (Jan-Aug 2018)",
).add_to(map_choropleth)

# display the map
map_choropleth
```

- `geo_data` is a GeoJSON FeatureCollection containing the boundaries of each geographical area.
  - We convert the districts GeoDataFrame to a GeoJSON FeatureCollection with the `__geo_interface__` attribute.
- `data` is a Pandas Series containing the values that will be used to color-code each geographical area.
- `key_on` will always be set to `feature.id`, based on the GeoJSON structure.
- `fill_color` sets the color scale.

## Manipulating Geospatial Data
*Find locations with just the name of a place. And, learn how to join data based on spatial relationships. [#](https://www.kaggle.com/alexisbcook/manipulating-geospatial-data)*

### Geocoding

Geocoding is the process of converting the name of a place or an address to a location on a map. We'll use `geopandas.tools.geocode()` to do all of our geocoding.

```python
from geopandas.tools import geocode
geocode("The Great Pyramid of Giza", provider="nominatim")
```

To use the geocoder, we need:

- the `name` or `address` as a Python string, and
- the name of the `provider`. To avoid having to provide an API key, we used the [OpenStreetMap Nominatim geocoder](https://nominatim.openstreetmap.org/).

It's often the case that we'll need to geocode many different addresses.

```python
# load Starbucks locations in California
import pandas as pd
starbucks = pd.read_csv("../input/geospatial-learn-course-data/starbucks_locations.csv")
```

```python
# define geocoder function
def my_geocoder(row):
    try:
        point = geocode(row, provider="nominatim")["geometry"][0]
        return pd.Series({"Latitude": point.y, "Longitude": point.x})
    except:
        return None
```

If the geocoding is successful, it returns a GeoDataFrame with two columns:

- the `geometry` column, which is a `Point` object, and we can get the `Latitude` and `Longitude` from the `y` and `x` attributes, respectively.
- the `address` column contains the full address.

When geocoding a large dataframe, you might encounter an error when geocoding.

- In case you get a time out error, try first using the `timeout` parameter (allow the service a bit more time to respond).
- In case of Too Many Requests error, you have hit the rate-limit of the service, and you should slow down your requests. `GeoPy` provides additional tools for taking into account rate limits in geocoding services. [More info](https://automating-gis-processes.github.io/site/notebooks/L3/geocoding_in_geopandas.html).

```python
# rows with missing locations
rows_with_missing = starbucks[starbucks["Latitude"].isna() | starbucks["Longitude"].isna()]
```

```python
# fill missing geo data
rows_with_missing = rows_with_missing.apply(lambda x: my_geocoder(x["Address"]), axis=1)

# drop rows that were not successfully geocoded
rows_with_missing.dropna(axis=0, subset=["Latitude", "Longitude"])

# update main DataFrame
starbucks.update(rows_with_missing)
```

### Table Joins

We can combine data from different sources.

#### Attribute Join

You already know how to use `pd.DataFrame.join()` to combine information from multiple DataFrames with a shared index. We refer to this way of joining data (by simpling matching values in the index) as an attribute join. We'll work with some DataFrames containing data and a unique id (in the `GEOID` column) for each county in the state of California.

```python
# create DataFrame contains an estimate of the population of each county
CA_pop = pd.read_csv(
    "../input/geospatial-learn-course-data/CA_county_population.csv", index_col="GEOID"
)
# create DataFrame contains the number of households with high income
CA_high_earners = pd.read_csv(
    "../input/geospatial-learn-course-data/CA_county_high_earners.csv", index_col="GEOID"
)
# create DataFrame contains the median age for each county
CA_median_age = pd.read_csv(
    "../input/geospatial-learn-course-data/CA_county_median_age.csv", index_col="GEOID"
)
```

```python
# use an attribute join
cols_to_add = CA_pop.join([CA_high_earners, CA_median_age]).reset_index()
```

When performing an attribute join with a GeoDataFrame, it's best to use the `gpd.GeoDataFrame.merge()`. We'll work with a GeoDataFrame `CA_counties` containing the name, area (in square kilometers), and a unique id (in the `GEOID` column) for each county in the state of California. The `geometry` column contains a polygon with county boundaries.

```python
import geopandas as gpd
CA_counties = gpd.read_file(
    "../input/geospatial-learn-course-data/CA_county_boundaries/CA_county_boundaries/CA_county_boundaries.shp"
)
```

```python
# use an attribute join
CA_stats = CA_counties.merge(cols_to_add, on="GEOID")
```

- The `on` argument is set to the column name that is used to match rows.

Now that we have all of the data in one place, it's much easier to calculate statistics that use a combination of columns.

```python
CA_stats["density"] = CA_stats["population"] / CA_stats["area_sqkm"]
```

#### Spatial Join

With a spatial join, we combine GeoDataFrames based on the spatial relationship between the objects in the `geometry` columns. We do this with `gpd.sjoin()`.

So, which counties look promising for **new** Starbucks Reserve Roastery?

```python
sel_counties = CA_stats[
    (CA_stats["high_earners"] >= 100000)
    & (CA_stats["median_age"] <= 38.5)
    & (CA_stats["density"] >= 285)
]
sel_counties.crs = {"init": "epsg:4326"}
```

```python
starbucks_gdf = gpd.GeoDataFrame(
    starbucks, geometry=gpd.points_from_xy(starbucks["Longitude"], starbucks["Latitude"])
)
starbucks_gdf.crs = {"init": "epsg:4326"}
```

```python
sel_counties_stores = gpd.sjoin(starbucks_gdf, sel_counties)
```

The spatial join above looks at the `geometry` columns in both GeoDataFrames. If a Point object from the `starbucks_gdf` GeoDataFrame intersects a Polygon object from the `sel_counties` DataFrame, the corresponding rows are combined and added as a single row of the `sel_counties_stores` DataFrame. Otherwise, counties without a matching starbuckses (and starbuckses without a matching county) are omitted from the results.

The `gpd.sjoin()` method is customizable for different types of joins, through the `how` and `op` arguments. For example, you can do the equivalent of a SQL left (or right) join by setting `how='left'` (or `how='right'`).

Let's visualize!

```python
# define the base map
from folium import Map
map_cluser = Map(location=[37, -120], zoom_start=6)

# add points to the map
from folium.plugins import MarkerCluster
mc = MarkerCluster()
from folium import Marker
for idx, row in sel_counties_stores.iterrows():
    mc.add_child(Marker([row["Latitude"], row["Longitude"]]))

map_cluser.add_child(mc)

# display the map
map_cluser
```

## Proximity Analysis
*Measure distance, and explore neighboring points on a map. [#](https://www.kaggle.com/alexisbcook/proximity-analysis)*

### Introduction

We'll explore several techniques for proximity analysis, such as:

- Measuring the distance between points on a map, and
- Selecting all points within some radius of a feature.

We want to identify how hospitals have been responding to crash collisions in New York City. We'll work with GeoDataFrame `collisions` tracking major motor vehicle collisions in 2013-2018.

```python
import geopandas as gpd
collisions = gpd.read_file(
    "../input/geospatial-learn-course-data/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions.shp"
)
hospitals = gpd.read_file(
    "../input/geospatial-learn-course-data/nyu_2451_34494/nyu_2451_34494/nyu_2451_34494.shp"
)
```

### Measuring Distance

To measure distances between points from two different GeoDataFrames, we first have to make sure that they use the same CRS.

```python
collisions.crs == hospitals.crs
```

- We also check the CRS to see which units it uses (meters, feet, or something else). In this case, EPSG 2263 has units of meters.

Then, we use the `distance()` method, returns a `Series` containing the distance to the others.

```python
# measure distance from a relatively recent collision to each hospital
distances = hospitals["geometry"].distance(collisions.iloc[-1]["geometry"])
```

```python
# calculate mean distance to hospitals
distances.mean()

# find the closest hospital
hospitals.iloc[distances.idxmin()][["ADDRESS", "LATITUDE", "LONGITUDE"]]
```

### Creating a Buffer

If we want to understand all points on a map that are some radius away from a point, the simplest way is to create a buffer. It's a `GeoSeries` containing multiple `Polygon` objects. Each polygon is a buffer around a different spot.

We'll create a DataFrame `outside_range` containing all rows from `collisions` with crashes that occurred more than 10 kilometers from the closest hospital.

```python
# create a GeoSeries buffer
ten_km_buffer = hospitals["geometry"].buffer(10000)
```

To test if a collision occurred within 10 kilometers of any hospital, we could run different tests for each polygon. But a more efficient way is to first collapse all of the polygons into a `MultiPolygon` object. We do this with the `unary_union` attribute.

```python
# turn group of polygons into single multipolygon
my_union = ten_km_buffer.unary_union
```

We use the `contains()` method to check if the multipolygon contains a point.

```python
# is the closest station less than two miles away?
my_union.contains(collisions.iloc[-1]["geometry"])
```

```python
outside_range = collisions.loc[~collisions["geometry"].apply(lambda x: my_union.contains(x))]
```

```python
# calculate the percentage of collisions occurred more than 10 km away from the closest hospital
round(100 * len(outside_range)/len(collisions), 2)
```

### Make a Recommender

When collisions occur in distant locations, it becomes even more vital that injured persons are transported to the nearest available hospital.

With this in mind, we want to create a recommender that:

- takes the location of the crash as input,
- finds the closest hospital, and
- returns the name of the closest hospital.

```python
def best_hospital(collision_location):
    idx_min = hospitals["geometry"].distance(collision_location).idxmin()
    return hospitals.iloc[idx_min]["name"]
```

```python
# suggest the closest hospital to the last collision
best_hospital(outside_range["geometry"].iloc[-1])
```

```python
# which hospital is most recommended?
outside_range["geometry"].apply(best_hospital).value_counts().idxmax()
```

Where should the city construct new hospitals? Lets visualize!

```python
# define the base map
from folium import Map
m = Map(location=[40.7, -74], zoom_start=11)

# add buffers' Polygon
from folium import GeoJson
GeoJson(ten_km_buffer.to_crs(epsg=4326)).add_to(m)

# add the heatmap of collisions, out of 10km buffers
from folium.plugins import HeatMap
HeatMap(data=outside_range[["LATITUDE", "LONGITUDE"]], radius=9).add_to(m)

# add (Lat,Long) popup
from folium import LatLngPopup
LatLngPopup().add_to(m)

# display the map
m
```

- We use `folium.GeoJson()` to plot each `Polygon` on a map. Note that since folium requires coordinates in latitude and longitude, we have to convert the CRS to EPSG 4326 before plotting.

# Intro to Machine Learning
*Learn the core ideas in machine learning, and build your first models.*

## How Models Work
*The first step if you're new to machine learning. [#](https://www.kaggle.com/dansbecker/how-models-work)*

- **Fitting** or **Training**: Capturing patterns from **training data**
- **Predicting**: Getting results from applying the model to **new data**

## Basic Data Exploration
*Load and understand your data. [#](https://www.kaggle.com/dansbecker/basic-data-exploration)*

### Get Familiar with the Data

```python
# load data
import pandas as pd
melbourne_data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
```

```python
# summary
melbourne_data.head()
melbourne_data.describe()
```

## Your First Machine Learning Model
*Building your first model. Hurray! [#](https://www.kaggle.com/dansbecker/your-first-machine-learning-model)*

### Selecting Data for Modeling

```python
# filter rows with missing values
dropna_melbourne_data = melbourne_data.dropna(axis=0)
```

```python
# separate target (y) from features (predictors, X)
y = dropna_melbourne_data["Price"]
feature_list = [
    "Rooms",
    "Bathroom",
    "Landsize",
    "BuildingArea",
    "YearBuilt",
    "Lattitude",
    "Longtitude",
]
X = dropna_melbourne_data[feature_list]
```

### Building the Model

- **Define**: What type of model will it be? A decision tree? Some other type of model?
- **Fit**: Capture patterns from provided data. This is the heart of modeling.
- **Predict**: Just what it sounds like.
- **Evaluate**: Determine how accurate the model's predictions are.

#### Decision Tree

```python
# define model
from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor(random_state=1)

# fit model
melbourne_model.fit(X, y)

# make prediction
predictions = melbourne_model.predict(X)
```

## Model Validation
*Measure the performance of your model? So you can test and compare alternatives. [#](https://www.kaggle.com/dansbecker/model-validation)*

### Summarizing the Model Quality into Metrics

There are many metrics for summarizing the model quality. **Predictive accuracy** means will the model's predictions be close to what actually happens?

**Mean Absolute Error** (MAE)

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, predictions)
```

**Big Mistake**: Measuring scores with the training data or the problem with **in-sample** scores!

### Validation Data
*Making predictions on **new** data*

Exclude some data from the model-building process, and then use those to test the model's accuracy.

```python
# break off validation set from training data, for both features and target
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
```

```python
# define model
melbourne_model = DecisionTreeRegressor(random_state=1)

# fit model
melbourne_model.fit(X_train, y_train)

# make prediction on validation data
predictions_val = melbourne_model.predict(X_valid)

# evaluate the model
mean_absolute_error(y_valid, predictions_val)
```

There are many ways to improve a model, such as

- Finding **better features**, the iterating process of building models with different features and comparing them to each other
- Finding **better model types**

## Underfitting and Overfitting
*Fine-tune your model for better performance. [#](https://www.kaggle.com/dansbecker/underfitting-and-overfitting)*

- **Overfitting**: Capturing spurious patterns that won't recur in the future, leading to less accurate predictions.
- **Underfitting**: Failing to capture relevant patterns, again leading to less accurate predictions.

In the **Decision Tree** model, the most important option to control the accuracy is the tree's **depth**, a measure of how many splits it makes before coming to a prediction.

- A **deep** tree makes leaves with fewer objects. It causes **overfitting**.
- A **shallow** tree makes big groups. It causes **underfitting**.

There are a few options for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes. But the `max_leaf_nodes` argument provides a very sensible way to control overfitting vs underfitting.

```python
# function for comparing MAE with differing values of max_leaf_nodes
def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    predictions_val = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, predictions_val)
    return(mae)
```

```python
# compare models
max_leaf_nodes_candidates = [5, 50, 500, 5000]
scores = {
    leaf_size: get_mae(leaf_size, X_train, X_valid, y_train, y_valid)
    for leaf_size in max_leaf_nodes_candidates
}
best_tree_size = min(scores, key=scores.get)
```

## Random Forests
*Using a more sophisticated machine learning algorithm. [#](https://www.kaggle.com/dansbecker/random-forests)*

### Introduction

Decision trees leave you with a difficult decision. A deep tree and overfitting vs. a shallow one and underfitting.

### Random Forest

A Random Forest model uses many trees, and makes a prediction by averaging the predictions of each component. It generally has much better predictive accuracy even with than a single decision tree, even with default parameters, without tuning the parameters like `max_leaf_nodes`.

```python
# define & fit model
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)

# make prediction
preds_valid = forest_model.predict(X_valid)

# evaluate the model
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_valid, preds_valid)
```

Some models, like the **XGBoost** model, provides better performance when tuned well with the right parameters (but which requires some skill to get the right model parameters).

## Exercise: Machine Learning Competitions
*Enter the world of machine learning competitions to keep improving and see your progress. [#](https://www.kaggle.com/kernels/fork/1259198)*

### Setup

```python
# load data
import pandas as pd
X_full = pd.read_csv("../input/train.csv", index_col="Id")
X_test_full = pd.read_csv("../input/test.csv", index_col="Id")
```

```python
# separate target (y) from features (X)
y = X_full["SalePrice"]
features = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
]
X = X_full[features].copy()
X_test = X_test_full[features].copy()
```

```python
# break off validation set from training data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
```

### Evaluate Several Models

```python
# define models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model_1 = DecisionTreeRegressor(random_state=0)
model_2 = DecisionTreeRegressor(max_leaf_nodes=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=50, random_state=0)
model_4 = RandomForestRegressor(n_estimators=100, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, criterion="mae", random_state=0)
model_6 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_7 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7]
```

```python
# function for comparing different models
from sklearn.metrics import mean_absolute_error
def score_model(model, X_train, X_valid, y_train, y_valid):
    # fit model
    model.fit(X_train, y_train)
    # make validation predictions
    preds_valid = model.predict(X_valid)
    # return mae
    return mean_absolute_error(y_valid, preds_valid)
```

```python
# compare models
for i in range(len(models)):
    mae = score_model(models[i])
    print(f"Model {i+1} MAE: {mae:,.0f}")
```

```bash
Model 1 MAE: 29,653
Model 2 MAE: 27,283
Model 3 MAE: 24,015
Model 4 MAE: 23,740
Model 5 MAE: 23,528
Model 6 MAE: 23,996
Model 7 MAE: 23,706
```

### Generate Test Predictions

```python
# define model, based on the most accurate model
my_model = RandomForestRegressor(n_estimators=100, criterion="mae", random_state=0)

# fit the model to the training data, all of it
my_model.fit(X, y)

# make test prediction
preds_test = my_model.predict(X_test)
```

```python
# save predictions in format used for competition scoring
output = pd.DataFrame({"Id": X_test.index, "SalePrice": preds_test})
output.to_csv("submission.csv", index=False)
```

# Intermediate Machine Learning
*Learn to handle missing values, non-numeric values, data leakage and more. Your models will be more accurate and useful.*

## Introduction
*Review what you need for this Micro-Course. [#](https://www.kaggle.com/alexisbcook/introduction)*

In this micro-course, you will accelerate your machine learning expertise by learning how to:

* Tackle data types often found in real-world datasets (**missing values**, **categorical variables**),
* Design **pipelines** to improve the quality of your machine learning code,
* Use advanced techniques for model validation (**cross-validation**),
* Build state-of-the-art models that are widely used to win Kaggle competitions (**XGBoost**), and
* Avoid common and important data science mistakes (**leakage**).

## Missing Values
*Missing values happen. Be prepared for this common challenge in real datasets. [#](https://www.kaggle.com/alexisbcook/missing-values)*

### Introduction
There are many ways data can end up with missing values. For example,

- A 2 bedroom house won't include a value for the size of a third bedroom.
- A survey respondent may choose not to share his income.

Most machine learning libraries (including scikit-learn) give an error if you try to build a model using data with missing values.

```python
# show number of missing values in each column
def missing_val_count(data):
    missing_val_count_by_column = data.isna().sum()
    return missing_val_count_by_column[missing_val_count_by_column > 0]
```

### Approaches

- Drop Columns with Missing Values
- Imputation

#### Setup

```python
# load data
import pandas as pd
X_full = pd.read_csv("../input/train.csv", index_col="Id")
X_test_full = pd.read_csv("../input/test.csv", index_col="Id")
```

```python
# remove rows with missing "SalePrice"
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
```

```python
# separate target (y) from features (X)
y = X_full["SalePrice"]
X_full.drop(["SalePrice"], axis=1, inplace=True)
```

```python
# use only numerical features, to keep things simple
X = X_full.select_dtypes(exclude=["object"])
X_test = X_test_full.select_dtypes(exclude=["object"])
```

```python
# break off validation set from training data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
```

```python
# get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isna().any()]
```

```python
# function for comparing different approaches
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds_valid = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds_valid)
```

#### Drop Columns with Missing Values

The model loses access to a lot of (potentially useful!) information with this approach.

```python
# drop `cols_with_missing` in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# evaluate the model
score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)
```

#### Imputation

Imputation fills in the missing values with some number.

```python
# imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))

# imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# evaluate the model
score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)
```

Strategy
- default=`mean` replaces missing values using the mean along each column. (only numeric)
- `median` replaces missing values using the median along each column. (only numeric)
- `most_frequent` replaces missing using the most frequent value along each column. (strings or numeric)
- `constant` replaces missing values with `fill_value`. (strings or numeric)

### Train and Evaluate Model

```python
# define and fit model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(imputed_X_train, y_train)

# make validation prediction
preds_valid = model.predict(imputed_X_valid)
mean_absolute_error(y_valid, preds_valid)
```

### Test Data

```python
# preprocess test data
imputed_X_test = pd.DataFrame(imputer.fit_transform(X_test))
# put column names back
imputed_X_test.columns = X_test.columns

# make test prediction
preds_test = model.predict(imputed_X_test)
```

```python
# save test predictions to file
output = pd.DataFrame({"Id": X_test.index, "SalePrice": preds_test})
output.to_csv("submission.csv", index=False)
```

## Categorical Variables
*There's a lot of non-numeric data out there. Here's how to use it for machine learning. [#](https://www.kaggle.com/alexisbcook/categorical-variables)*

### Introduction

A categorical variable takes only a limited number of values.
- **Ordinal**: A question that asks "how often you eat breakfast?" and provides four options: "Never", "Rarely", "Most days", or "Every day".
- **Nominal**: A question that asks "what brand of car you own?".

Most machine learning libraries (including scikit-learn) give an error if you try to build a model using data with categorical variables.

### Approaches

- Drop Categorical Variables
- Label Encoding
- One-Hot Encoding

#### Setup

```python
# load data
import pandas as pd
X_full = pd.read_csv("../input/train.csv", index_col="Id")
X_test_full = pd.read_csv("../input/test.csv", index_col="Id")
```

```python
# remove rows with missing target
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
```

```python
# separate target (y) from features (X)
y = data["Price"]
X = data.drop(["Price"], axis=1)
```

```python
# break off validation set from training data
from sklearn.model_selection import train_test_split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
```

```python
# handle missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isna().any()]
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)
```

```python
# select categorical columns with relatively low cardinality, to keep things simple
# cardinality means the number of unique values in a column
categorical_cols = [
    cname
    for cname in X_train_full.columns
    if (X_train_full[cname].dtype == "object") and (X_train_full[cname].nunique() < 10)
]

# select numerical columns
numerical_cols = [
    cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]
]

# keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
```

```python
# function for comparing different approaches
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

#### Drop Categorical Variables
*This approach will only work well if the columns did not contain useful information.*

```python
# drop catagorial columns
drop_X_train = X_train.select_dtypes(exclude=["object"])
drop_X_valid = X_valid.select_dtypes(exclude=["object"])

# evaluate the model
score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
```

#### Label Encoding

Label encoding assigns each unique value, that appears in the training data, to a different integer.

In the case that the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them. It should be used only for **target labels encoding**.

To encode categorical features, use One-Hot Encoder, which can handle unseen values.

For **tree-based models** (like decision trees and random forests), you can expect label encoding to work well with **ordinal** variables.

```python
# find columns, which are in validation data but not in training data
good_label_cols = [col for col in categorical_cols if set(X_train[col]) == set(X_valid[col])]
bad_label_cols = list(set(categorical_cols) - set(good_label_cols))

# drop them
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# apply label encoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

# evaluate the model
score_dataset(label_X_train, label_X_valid, y_train, y_valid)
```

#### One-Hot Encoding

One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data. Useful parameters are:

  - `handle_unknown="ignore"` avoids errors when the validation data contains classes that aren't represented in the training data,
  - `sparse=False` returns the encoded columns as a numpy array (instead of a sparse matrix).

In contrast to label encoding, one-hot encoding does not assume an ordering of the categories. Thus, you can expect this approach to work particularly well with categorical variables without an intrinsic ranking, we refer them as **nominal** variables.

One-hot encoding generally does **not** perform well with high-cardinality categorical variable (i.e., more than 15 different values). **Cardinality** means the number of unique values in a column.

```python
# get cardinality for each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), categorical_cols))
d = dict(zip(categorical_cols, object_nunique))

# print cardinality by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
```

For this reason, we typically will only one-hot encode columns with relatively low cardinality. Then, high cardinality columns can either be dropped from the dataset, or we can use label encoding.

```python
# columns that will be one-hot encoded
low_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() < 10]

# columns that will be dropped from the dataset
high_cardinality_cols = list(set(categorical_cols) - set(low_cardinality_cols))

# apply one-hot encoder to each column with categorical data
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(X_train[low_cardinality_cols]))
oh_cols_valid = pd.DataFrame(oh_encoder.transform(X_valid[low_cardinality_cols]))

# one-hot encoding removed index; put it back
oh_cols_train.index = X_train.index
oh_cols_valid.index = X_valid.index

# drop all categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(categorical_cols, axis=1)
num_X_valid = X_valid.drop(categorical_cols, axis=1)

# add one-hot encoded columns to numerical features
oh_X_train = pd.concat([num_X_train, oh_cols_train], axis=1)
oh_X_valid = pd.concat([num_X_valid, oh_cols_valid], axis=1)

# evaluate the model
score_dataset(oh_X_train, oh_X_valid, y_train, y_valid)
```

Doing all things seperately for training, evaluating and testing is way DIFFICULT. Doing with Pipelines is **FUN**!

## Pipelines
*A critical skill for deploying (and even testing) complex models with pre-processing. [#](https://www.kaggle.com/alexisbcook/pipelines)*

### Introduction

Pipelines are a simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline **bundles preprocessing and modeling steps** so you can use the whole bundle as if it were a single step.

Some important benefits of pipelines are:

- **Cleaner Code**: Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step.
- **Fewer Bugs**: There are fewer opportunities to misapply a step or forget a preprocessing step.
- **Easier to Productionize**: It can be surprisingly hard to transition a model from a prototype to something deployable at scale, but pipelines can help.
- **More Options for Model Validation**: You will see an example in the Cross-Validation tutorial.

### Steps

- Setup
- Define Preprocessing Steps
- Define the Model
- Create and Evaluate the Pipeline

#### Setup

```python
# load data
import pandas as pd
X_full = pd.read_csv("../input/train.csv", index_col="Id")
X_test_full = pd.read_csv("../input/test.csv", index_col="Id")
```

```python
# remove rows with missing target
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
```

```python
# separate target (y) from features (X)
y = X_full["Price"]
X = X_full.drop(["Price"], axis=1)
```

```python
# break off validation set from training data
from sklearn.model_selection import train_test_split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
```

```python
# select categorical columns with relatively low cardinality, to keep things simple
# cardinality means the number of unique values in a column
categorical_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
]

# select numerical columns
numerical_cols = [
    cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]
]

# keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
```

#### Define Preprocessing Steps

Similar to how a pipeline bundles together preprocessing and modeling steps, we use the `ColumnTransformer` class to bundle together different preprocessing steps. The code below:

- Imputes missing values in numerical data, and
- Imputes missing values and applies a one-hot encoding to categorical data.

```python
from sklearn.pipeline import Pipeline

# preprocessing for numerical data
from sklearn.impute import SimpleImputer
numerical_transformer = SimpleImputer(strategy="most_frequent")

# preprocessing for categorical data
from sklearn.preprocessing import OneHotEncoder
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# bundle preprocessing for numerical and categorical data
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)
```

#### Define the Model

For example we use `RandomForestRegressor`

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, min_samples_split=3, random_state=0)
```

#### Create and Evaluate the Pipeline

We use the `Pipeline` class to define a pipeline that bundles the preprocessing and modeling steps.

- With the pipeline, we preprocess the training data and fit the model in a single line of code.
- With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline automatically preprocesses the features before generating predictions.

```python
# bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# evaluate the model
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_valid, preds)
```

### Test

```python
# preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# save predictions in format used for competition scoring
output = pd.DataFrame({"Id": X_test.index, "SalePrice": preds_test})
output.to_csv("submission.csv", index=False)
```

**Vola**!

## Cross-Validation
*A better way to test your models. [#](https://www.kaggle.com/alexisbcook/cross-validation)*

### Introduction

Machine learning is an **iterative process**. You will face choices about what **predictive variables** to use, what **types of models** to use, what **arguments** to supply to those models, etc.

In a dataset with 5000 rows, you will typically keep about 20% of the data as a validation dataset, or 1000 rows. But this leaves some random chance in determining model scores. That is, a model might do well on one set of 1000 rows, even if it would be inaccurate on a different 1000 rows.

The larger the validation set, the less randomness (aka "noise") there is in our measure of model quality.

### Cross-Validation

In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality.

In Experiment 1, we use the first **fold** (20%) as a **validation (or holdout) set** and everything else as training data. We repeat this process, using every fold once as the holdout set.

Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset.

Cross-validation gives a more accurate measure of model quality. However, it can take longer to run.

For **small datasets**, you should run cross-validation. But for larger datasets, a single validation set is sufficient.

There's no simple threshold for what constitutes a large vs. small dataset. But if your model takes a couple minutes or less to run, it's probably worth switching to cross-validation. Or you can run cross-validation and see if the scores for each experiment seem close.

### Steps

- Setup
- Define a Pipeline
- Obtain the Cross-validation Scores
- Combine them as a Function
- Evalute the Model Performance
- Find the best Parameter Value

#### Setup

```python
# load data
import pandas as pd
X_full = pd.read_csv("../input/train.csv", index_col="Id")
```

```python
# remove rows with missing target
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
```

```python
# separate target (y) from features (X)
y = X_full["SalePrice"]
X_full.drop(["SalePrice"], axis=1, inplace=True)
```

```python
# select numeric columns only
numeric_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ["int64", "float64"]]
X = X_full[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()
```

#### Define a Pipeline
*It's difficult to do cross-validation without pipelines.*

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
my_pipeline = Pipeline(
    steps=[
        ("preprocessor", SimpleImputer()),
        ("model", RandomForestRegressor(n_estimators=50, random_state=0)),
    ]
)
```

### Obtain the Cross-validation Scores
*with the `cross_val_score()` function*

```python
from sklearn.model_selection import cross_val_score
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
scores
>>> [301628 303164 287298 236061 260383]

# take the average score across experiments
scores.mean()
>>> 277707
```
- The `cv` parameter sets the number of folds.
- The `scoring` parameter chooses a measure of model quality to report. The docs for scikit-learn show a [list of options](http://scikit-learn.org/stable/modules/model_evaluation.html).
- It is a little surprising that we specify **negative MAE**. Scikit-learn has a convention where all metrics are defined so a high number is better. Using negatives here allows them to be consistent with that convention. So multiply this score by -1.

#### Combine them as a Function

```python
# get validation scores based on different numbers of estimators (trees)
def get_score(n_estimators):
    """return the average MAE over 3 CV folds of random forest model."""

    my_pipeline = Pipeline(
        steps=[
            ("preprocessor", SimpleImputer()),
            ("model", RandomForestRegressor(n_estimators=n_estimators, random_state=0)),
        ]
    )

    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring="neg_mean_absolute_error")

    return scores.mean()
```

#### Evalute the Model Performance
*Corresponding to some different values.*

```python
# for example 50, 100, 150, ..., 300, 350, 400
results = {i: get_score(i) for i in range(50, 450, 50)}
```

#### Find the best Parameter Value

```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(results.keys(), results.values())
plt.show()
```

If you'd like to learn more about **hyperparameter optimization**, you're encouraged to start with **grid search**, which is a straightforward method for determining the best combination of parameters for a machine learning model. Thankfully, scikit-learn also contains a built-in function `GridSearchCV()` that can make your grid search code very efficient!

## XGBoost
*The most accurate modeling technique for structured data. [#](https://www.kaggle.com/alexisbcook/xgboost)*

### Introduction

Ensemble methods combine the predictions of several models and achieve better performance, like Random Forest method.

### Gradient Boosting

It is a ensemble method that goes through cycles to iteratively add models into an ensemble. Steps are:

1. It begins by initializing the ensemble with a **naive model**, even if its predictions are wildly inaccurate.
2. Then it uses the all models in the current ensemble to **generate predictions** for each observation in the dataset.
3. These predictions are used to **calculate a loss function** (like mean squared error, for instance).
4. Then, it uses the loss function to **fit a new model** that will be added to the ensemble and **reduce the loss**. (It uses [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) on the loss function to determine the parameters in this new model.)
5. Then it adds the new model to ensemble.
6. ... **repeat** steps 2-5!

**XGBoost** stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional features focused on performance and speed. (Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages.)

### Steps

- Setup
- `XGBRegressor`
- Make Predictions
- Parameter Tuning

#### Setup

```python
# load data
import pandas as pd
X_full = pd.read_csv("../input/train.csv", index_col="Id")
X_test_full = pd.read_csv("../input/test.csv", index_col="Id")
```

```python
# remove rows with missing target
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
```
```python
# separate target (y) from features (X)
y = X_full["SalePrice"]
X = X_full.drop(["SalePrice"], axis=1, inplace=True)
```

```python
# break off validation set from training data
from sklearn.model_selection import train_test_split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
```

```python
# select categorical columns with relatively low cardinality (convenient but arbitrary)
# cardinality means the number of unique values in a column
low_cardinality_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
]

# select numeric columns
numeric_cols = [
    cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]
]

# keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
```

```python
# one-hot encode the data
# to shorten the code, we use pandas
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join="left", axis=1)
X_train, X_test = X_train.align(X_test, join="left", axis=1)
```

#### `XGBRegressor`
*The scikit-learn API for XGBoost.*

```python
# define model
from xgboost import XGBRegressor
my_model = XGBRegressor()

# fit model
my_model.fit(X_train, y_train)
```

#### Make Predictions

```python
# make predictions
predictions = my_model.predict(X_valid)

# evaluate the model
from sklearn.metrics import mean_absolute_error
mean_absolute_error(predictions, y_valid)
```

#### Parameter Tuning

```python
my_model = XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=4)
my_model.fit(
    X_train, y_train,
    early_stopping_rounds=5,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)
```

`n_estimators`

- It specifies how many times to go through the modeling cycle. It is equal to the number of models that we include in the ensemble.
- **Too low** a value causes **underfitting**.
- **Too high** a value causes **overfitting**.
- Typical values range from 100-1000, though this depends a lot on the `learning_rate` parameter.

`early_stopping_rounds`

- It offers a way to **automatically find the ideal value** for `n_estimators`.
- Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for `n_estimators`.
- Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping.
- Setting `early_stopping_rounds=5` is a reasonable choice.
- When using `early_stopping_rounds`, you also need to set aside some data for calculating the validation scores. This is done by setting the `eval_set` parameter.

`learning_rate`

- Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in. This means each tree we add to the ensemble helps us less. So, we can set a **higher** value for `n_estimators` **without overfitting**.
- If we use early stopping, the appropriate number of trees will be determined automatically.
- In general, a **small** learning rate and **large** number of estimators will yield **more accurate** XGBoost models, though it will also take the model **longer to train** since it does more iterations through the cycle. As default, XGBoost sets `learning_rate=0.1`.

`n_jobs`

- You can use parallelism to build your models faster. It's common to set the parameter `n_jobs` equal to the number of cores on your machine.
- On smaller datasets, this won't help. But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.


## Data Leakage
*Find and fix this problem that ruins your model in subtle ways. [#](https://www.kaggle.com/alexisbcook/data-leakage)*

### Introduction

Data leakage (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction.

This causes a model to **look accurate** until you start making decisions with the model, and then the model becomes very inaccurate.

There are two main types of leakage: target leakage and train-test contamination.

### Target Leakage

It occurs when your predictors include **data becomes available after predictions**.

Example: People take antibiotic medicines after getting pneumonia in order to recover.

- The data shows a strong relationship between those columns.
- But `took_antibiotic_medicine` is frequently changed after the value for `got_pneumonia` is determined.
- The model would see that anyone who has a value of `False` for `took_antibiotic_medicine` didn't have pneumonia.
- Since validation data comes from the same source as training data, the pattern will repeat itself in validation, and the model will have great validation (or cross-validation) scores.

**Prevent**

Any variable updated (or created) after the target value is realized should be excluded.

### Train-Test Contamination

It occurs when **the validation data affects the preprocessing behavior**.

Example: Imagine you run preprocessing (like fitting an imputer for missing values) before calling `train_test_split()`.

This problem becomes even more dangerous when you do more complex feature engineering.

**Prevent**

- If your validation is based on a simple train-test split, exclude the validation data from any type of fitting, including the fitting of preprocessing steps.
- This is easier if you use scikit-learn **pipelines**.
- When using cross-validation, it's even more critical that you do your preprocessing inside the pipeline!

### Examples

#### The Data Science of Shoelaces

Build a model to predict how many shoelaces NIKE needs each month.

The most important features in the model are

- The current month
- Advertising expenditures in the previous month
- Various macroeconomic features (like the unemployment rate) as of the beginning of the current month
- The amount of leather they ended up using in the current month

The results show the model is almost perfectly accurate if you include the feature about how much leather they used because the amount of leather they use is a **perfect indicator** of how many shoes they produce.

Is the leather used feature constitutes a source of data leakage?

**Solution**

- It depends on details of **how data is collected** (which is common when thinking about leakage).
- Would you at the beginning of the month decide how much leather will be used that month? If so, this is ok. But if that is determined during the month, you would not have access to it when you make the prediction.

You could use the amount of leather they ordered (rather than the amount they actually used) leading up to a given month as a predictor in your shoelace model.

Is this constitutes a source of data leakage?

**Solution**

- This could be fine, but it depends on whether they order shoelaces first or leather first.
- If they order shoelaces first, you won't know how much leather they've ordered when you predict their shoelace needs.

#### Getting Rich with Cryptocurrencies

Build a model to predict the price of a new cryptocurrency one day ahead.

The most important features in the model are

- Current price of the currency
- Amount of the currency sold in the last 24 hours
- Change in the currency price in the last 24 hours
- Change in the currency price in the last 1 hour
- Number of new tweets in the last 24 hours that mention the currency

The value of the cryptocurrency in dollars has fluctuated up and down by over 100$ in the last year, and yet the model′s average error is less than 1$.

Do you invest based on this model?

**Solution**

- There is no source of leakage here. These features should be available at the moment you want to make a predition, and they're unlikely to be changed in the training data after the prediction target is determined.
- But, this model's accuracy could be misleading if you aren't careful.
- If the price moves gradually, today's price will be an accurate predictor of tomorrow's price, but it may not tell you whether it's a good time to invest.
- A better prediction target would be the change in price (up or down and by how much) over the next day.

#### Housing Prices

Build a model to predict housing prices.

The most important features in the model are

- Size of the house (in square meters)
- Average sales price of homes in the same neighborhood
- Latitude and longitude of the house
- Whether the house has a basement

Which of the features is most likely to be a source of leakage?

**Solution**:

- Average sales price of homes in the same neighborhood is the source of target leakage.
  - We don't know the rules for when this is updated.
  - If the field is updated in the raw data after a home was sold, and the home's sale is used to calculate the average, this constitutes a case of target leakage.
  - At an extreme, if only one home is sold in the neighborhood, and it is the home we are trying to predict, then the average will be exactly equal to the value we are trying to predict.
  - In general, for neighborhoods with few sales, the model will perform very well on the training data. But when you apply the model, the home you are predicting won't have been sold yet, so this feature won't work the same as it did in the training data.
- Other features don't change, and will be available at the time we want to make a prediction.

#### Credit Card Applications

Build a model to predict which applications were accepted.

```python
# load data
import pandas as pd
data = pd.read_csv(
    "../input/aer-credit-card-data/AER_credit_card_data.csv",
    true_values=["yes"],
    false_values=["no"],
)
```

```python
# separate target (y) from features (X)
y = data["card"]
X = data.drop(["card"], axis=1)
```

Since this is a small dataset, we will use cross-validation to ensure accurate measures of model quality.

```python
# since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))

# evalutae model
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(my_pipeline, X, y, cv=5, scoring="accuracy")
cv_scores.mean()
>>> 0.981043
```

With experience, you'll find that it's very rare to find models that are accurate 98% of the time. It happens, but it's uncommon enough that we should inspect the data more closely for **target leakage**.

Summary of the data:

- card: 1 if credit card application accepted, 0 if not
- reports: Number of major derogatory reports
- age: Age n years plus twelfths of a year
- income: Yearly income (divided by 10,000)
- share: Ratio of monthly credit card expenditure to yearly income
- expenditure: Average monthly credit card expenditure
- owner: 1 if owns home, 0 if rents
- selfempl: 1 if self-employed, 0 if not
- dependents: 1 + number of dependents
- months: Months living at current address
- majorcards: Number of major credit cards held
- active: Number of active credit accounts

A few variables look suspicious. For example, does `expenditure` mean expenditure on this card or on cards used before appying?

```python
# fraction of those who received a card and had no expenditures
(X["expenditure"][y] == 0).mean()
>>> 0.02
```

```python
# fraction of those who did not receive a card and had no expenditures
(X["expenditure"][~y] == 0).mean()
>>> 1.00
```

As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures. This seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.

Since `share` is partially determined by `expenditure`, it should be excluded too.

The variables `active` and `majorcards` are a little less clear, but from the description, they sound concerning.

```python
# drop leaky features from dataset
potential_leaks = ["expenditure", "share", "active", "majorcards"]
X2 = X.drop(potential_leaks, axis=1)

# evaluate the model, with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, cv=5, scoring="accuracy")
cv_scores.mean()
>>> 0.831679
```

# Feature Engineering
*Discover the most effective way to improve your models.*

In this micro-course, you will learn a practical approach to feature engineering:

- Develop a baseline model for comparing performance on models with more features
- Encode categorical features so the model can make better use of the information
- Generate new features to provide more information for the model
- Select features to reduce overfitting and increase prediction speed

## Baseline Model
*Building a baseline model as a starting point for feature engineering. [#](https://www.kaggle.com/matleonard/baseline-model)*

### Introduction

We'll apply the techniques using data from Kickstarter projects. What we can do here is predict if a Kickstarter project will succeed. We get the outcome from the `state` column. To predict the outcome we can use features such as category, currency, funding goal, country, and when it was launched.

```python
import pandas as pd
ks = pd.read_csv(
    "../input/kickstarter-projects/ks-projects-201801.csv", parse_dates=["deadline", "launched"]
)
```

### Preparing Target Column

We'll convert the column into something can be used as targets in a model.

```python
# look at project states
ks["state"].unique()

# how many records of each?
ks.groupby("state")["ID"].count()
```

```python
# drop live projects
ks = ks.query('state != "live"')

# add outcome column, "successful" = 1, others are 0
ks = ks.assign(outcome=(ks["state"] == "successful").astype(int))
```

### Converting Timestamps

We can convert timestamp features into numeric catagorical features we can use in a model. If we loaded a column as timestamp data, we access date and time values through the `.dt` attribute on that column.

```python
ks = ks.assign(
    hour=ks["launched"].dt.hour,
    day=ks["launched"].dt.day,
    month=ks["launched"].dt.month,
    year=ks["launched"].dt.year,
)
```

### Preparing Categorical Variables

Now for the categorical variables (`category`, `currency`, and `country`), we'll need to convert them into integers so our model can use the data. For this we'll use scikit-learn's `LabelEncoder`.

- One-Hot encoding in high cardinality features will create an extremely **sparse** matrix, and will make your model run very slow, so in general you want to avoid one-hot encoding features with many levels. In addition, LightGBM models work with label encoded features, so you don't actually need to one-hot encode the categorical features.

```python
# choose catagorical features
cat_features = ["category", "currency", "country"]
```

```python
# create the encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# encode catagorical features
encoded = ks[cat_features].apply(le.fit_transform)
```

```python
# join `ks` and `encoded`. since they have the same index, we can easily join them
data = ks[["goal", "hour", "day", "month", "year", "outcome"]].join(encoded)
```

### Creating Training, Validation, and Test Splits

This is a **timeseries** dataset. Is there any special consideration when creating train/test splits for timeseries?

- Since our model is meant to predict events in the future, we must also validate the model on events in the future. If the data is mixed up between the training and test sets, then future data will **leak** in to the model and our validation results will overestimate the performance on new data.

We'll use a fairly simple approach and split the data using slices. We'll use 10% of the data as a validation set, 10% for testing, and the other 80% for training.

```python
# split data
valid_fraction = 0.1
valid_size = int(len(ks) * valid_fraction)
train = data[: -2 * valid_size]
valid = data[-2 * valid_size : -valid_size]
test = data[-valid_size:]
```

In general we want to be careful that each data set has the same proportion of target classes. A good way to do this automatically is with `sklearn.model_selection.StratifiedShuffleSplit`.

```python
# test of target classes distribution
for each in [train, valid, test]:
    # because outcome is 1 or 0
    print(each["outcome"].mean())
```

### Training a LightGBM Model

For this course we'll be using a LightGBM model. This is a tree-based model that typically provides the best performance, even compared to XGBoost. It's also relatively fast to train. We won't do hyperparameter optimization because that isn't the goal of this course. So, our models won't be the absolute best performance you can get. But you'll still see model performance improve as we do feature engineering.

```python
# define features
feature_cols = train.columns.drop("outcome")

# define train and valid datasets
import lightgbm as lgb
train_data = lgb.Dataset(train[feature_cols], label=train["outcome"])
valid_data = lgb.Dataset(valid[feature_cols], label=valid["outcome"])
```

```python
# fit model
param = {"num_leaves": 64, "objective": "binary", "metric": ["auc"], "seed": 7}
bst = lgb.train(
    param,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    early_stopping_rounds=10,
    verbose_eval=False,
)
```

- `metric`(s) will be used to evaluate on the evaluation set(s).
  - `auc` stands for [Area Under the Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).
- If you have a validation set, you can use early stopping to find the optimal number of boosting rounds. Early stopping requires at least one set in `valid_sets`. If there is more than one, it will use all of them except the training data. The model will train until the validation score stops improving. Validation score needs to improve at least every `early_stopping_rounds` to continue training.
- More info about [Python Quick Start](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html), [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) & [Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

### Making Predictions

```python
# make predictions
y_pred = bst.predict(test[feature_cols])
```

```python
# evalutae the model, based on metric(s)
from sklearn.metrics import roc_auc_score
roc_auc_score(test["outcome"], y_pred)
>>> 0.7476
```

## Categorical Encodings
*There are many ways to encode categorical data for modeling. Some are pretty clever. [#](https://www.kaggle.com/matleonard/categorical-encodings)*

### Introduction

Now that we've built a baseline model, we are ready to improve it with some clever ways to convert categorical variables into numerical features. The most basic encodings are one-hot encoding and label encoding. Here, we'll learn count encoding, target encoding (and variations), and singular value decomposition. We'll use the `categorical-encodings` package to get these encodings. These encoders work like scikit-learn transformers with `.fit` and `.transform` methods. We'll compare our encodings with the baseline model from the first tutorial.

```python
# function that splits data
def get_data_splits(data, valid_fraction=0.1):
    valid_size = int(len(data) * valid_fraction)
    train = data[: -valid_size * 2]
    valid = data[-valid_size * 2 : -valid_size]
    test = data[-valid_size:]
    return train, valid, test
```

```python
# function that trains a model
def train_model(train, valid, test=None, feature_cols=None):
    # define features
    if feature_cols is None:
        feature_cols = train.columns.drop("outcome")

    # define train and valid datasets
    import lightgbm as lgb
    train_data = lgb.Dataset(train[feature_cols], label=train["outcome"])
    valid_data = lgb.Dataset(valid[feature_cols], label=valid["outcome"])

    # fit model
    param = {"num_leaves": 64, "objective": "binary", "metric": ["auc"], "seed": 7}
    bst = lgb.train(
        param,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    # make predictions
    valid_pred = bst.predict(valid[feature_cols])

    # evaluate the model
    from sklearn.metrics import roc_auc_score
    valid_score = roc_auc_score(valid["outcome"], valid_pred)

    if test is not None:
        test_pred = bst.predict(test[feature_cols])
        test_score = roc_auc_score(test["outcome"], test_pred)
        return bst, valid_score, test_score
    else:
        return bst, valid_score
```

```python
# split data
train, valid, test = get_data_splits(baseline_data)

# train a model on the baseline data
bst = train_model(train, valid)
>>> 0.7467
```

### Count Encoding, Target Encoding & CatBoost Encoding

`CountEncoder` replaces each categorical value with the number of times it appears in the dataset. So, rare values tend to have similar counts (with values like 1 or 2), so we can classify rare values together at prediction time. Common values with large counts are unlikely to have the same exact count as other values. So, the common/important values get their own grouping.

`TargetEncoder` replaces a categorical value with the average value of the target for that value of the feature. This is often blended with the target probability over the entire dataset to reduce the variance of values with few occurences. This technique uses the targets to create new features. So including the validation or test data in the target encodings would be a form of **target leakage**. Instead, we should learn the target encodings from the training dataset only and apply it to the other datasets.

`CatBoostEncoder`, similar to `TargetEncoder`, is based on the target probablity for a given value. However with CatBoost, for each row, the target probability is calculated only from the rows before it.

```python
# split data
train, valid, test = get_data_splits(baseline_data)
```

```python
# choose catagorical features
cat_features = ["category", "currency", "country"]
```

```python
# create the encoder
from category_encoders import CountEncoder
ce = CountEncoder(cols=cat_features)
```

- `TargetEncoder`, `CatBoostEncoder` are the same.

```python
# learn encoding from the training features
ce.fit(train[cat_features])
```

- `TargetEncoder`, `CatBoostEncoder` are the same, but also target must be set in `fit()` method too. For instance `te.fit(train[cat_features], train["outcome"])`.

```python
# apply encoding to the train and validation sets as new columns with a suffix
train_encoded = train.join(ce.transform(train[cat_features]).add_suffix("_ce"))
valid_encoded = valid.join(ce.transform(valid[cat_features]).add_suffix("_ce"))
```

```python
# fit & evaluate a model on the encoded data
bst = train_model(train_encoded, valid_encoded)
```

We'll keep those work best. For instance `CatBoostEncoder` is that one.

```python
encoded = cb.transform(baseline_data[cat_features])
for col in encoded:
    baseline_data.insert(len(baseline_data.columns), col + "_cb", encoded[col])
```

## Feature Generation
*The frequently useful case where you can combine data from multiple rows into useful features. [#](https://www.kaggle.com/matleonard/feature-generation)*

### Introduction

Creating new features from the raw data is one of the best ways to improve your model. The features you create are different for every dataset, so it takes a bit of creativity and experimentation.

```python
# load data
import pandas as pd
ks = pd.read_csv(
    "../input/kickstarter-projects/ks-projects-201801.csv", parse_dates=["deadline", "launched"]
)

# drop live projects
ks = ks.query('state != "live"')

# add outcome column, "successful" = 1, others are 0
ks = ks.assign(outcome=(ks["state"] == "successful").astype(int))

# timestamp features
ks = ks.assign(
    hour=ks["launched"].dt.hour,
    day=ks["launched"].dt.day,
    month=ks["launched"].dt.month,
    year=ks["launched"].dt.year,
)

# choose catagorical features
cat_features = ["category", "currency", "country"]

# create the encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# encode catagorical features
encoded = ks[cat_features].apply(le.fit_transform)

# join `ks` and `encoded`
baseline_data = ks[["goal", "hour", "day", "month", "year", "outcome"]].join(encoded)
```

### Interactions

One of the easiest ways to create new features is by combining categorical variables. For example, if one record has the country `CA` and category `Music`, you can create a new value `CA_Music`. This is a new categorical feature that can provide information about **correlations** between categorical variables. This type of feature is typically called an interaction.

The easiest way to iterate through the pairs of features is with `itertools.combinations`.

```python
# choose catagorical features
cat_features = ["category", "currency", "country"]
```

```python
# create an empty DataFrame with same index
interactions = pd.DataFrame(index=ks.index)

# iterate through each pair of features & combine them
from itertools import combinations
for cname1, cname2 in combinations(cat_features, 2):
    # create new cname
    new_cname = "_".join([cname1, cname2])

    # convert to strings and combine
    new_values = ks[cname1].map(str) + "_" + ks[cname2].map(str)

    # encode the interaction feature
    le = LabelEncoder()
    interactions[new_cname] = le.fit_transform(new_values)
```

```python
# join interaction features to the data
baseline_data = baseline_data.join(interactions)
```

### Number of Projects in the Last Week

We'll count the number of projects launched in the preceeding week for each record. With a timeseries index, you can use `.rolling` to select time periods as the **window**. For example `.rolling('7d')` creates a rolling window that contains all the data in the previous 7 days. The window contains the current record, so if we want to count all the previous projects but not the current one, we'll need to subtract 1.

We'll use the `.rolling` method on a series with the `launched` column as the index. We'll create the series, using `ks["launched"]` as the index and `ks.index` as the values, then sort the times. Using a time series as the index allows us to define the rolling window size in terms of hours, days, weeks, etc.

```python
# function takes a timestamps Series, returns another Series with the number of events in time periods
def count_past_events(series, time_window):
    # create a Series with a timestamp index
    series = pd.Series(series.index, index=series).sort_index()
    
    # subtract 1 so the current event isn't counted
    past_events = series.rolling(time_window).count() - 1

    # adjust the index, so we can join it to the other training data
    past_events.index = series.values
    past_events = past_events.reindex(series.index)
    
    return past_events
```

```python
# count past events, in the previous 7 days
count_7_days = count_past_events(ks["launched"], time_window="7d")
```

```python
# join new features to the data
baseline_data.join(count_7_days)
```

### Time since the Last Project in the Same Category

Do projects in the same category compete for donors? We can capture this by calculating the time since the last launch project in the same category.

A handy method for performing operations within groups is to use `.groupby` then `.transform`. The `.transform` method takes a function then passes a series or dataframe to that function for each group. This returns a dataframe with the same indices as the original dataframe.

```python
# function calculates the time differences (in hours)
def time_since_last_project(series):
    return series.diff().dt.total_seconds() / 3600
```

```python
# calculate the time differences for each category
df = ks[["category", "launched"]].sort_values("launched")
timedeltas = df.groupby("category").transform(time_since_last_project)
```

We get `NaN`s here for projects that are the first in their category.

```python
timedeltas = timedeltas.fillna(timedeltas.median()).reindex(baseline_data.index)
```

```python
# join new features to the data
baseline_data.join(timedeltas)
```

### Transforming Numerical Features: Tree-based vs Neural Network Models

So far we've been using LightGBM, a tree-based model. Would these features we've generated work well for neural networks as well as tree-based models?

The features themselves will work for either model. However, numerical inputs to neural networks need to be standardized first. So, the features need to be scaled such that they have 0 mean and a standard deviation of 1. This can be done using `sklearn.preprocessing.StandardScaler`.

Some models work better when the features are **normally distributed**. Common choices for this are the **log** and **power** transformations. The log transformation won't help tree-based models because they are scale invariant. However, this should help if we had a linear model or neural network.

Other transformations include squares and other powers, exponentials, etc. These might help the model discriminate, like the kernel trick for SVMs. But again, it takes a bit of experimentation to see what works.

## Feature Selection
*You can make a lot of features. Here's how to get the best set of features for your model. [#](https://www.kaggle.com/matleonard/feature-selection)*

### Introduction

Now that we've generated hundreds or thousands of features after various encodings and feature generation, we should remove some of them to:

- Reduce the size of the model and speed up **model-training** and **hyperparameters-optimizing**, especially when building user-facing products.
- Prevent **overfitting** the model to the training and validation sets and improve the performance.

### Univariate Feature Selection

The simplest and fastest methods are based on univariate statistical tests. For each feature, measure how strongly the target depends on the feature using a statistical test like Ⲭ² or ANOVA.

From the `sklearn.feature_selection` module, `SelectKBest` returns the K-best features given some scoring function. For our classification problem, the module provides three different scoring functions: Ⲭ², ANOVA F-value, and the mutual information score.

- The F-value measures the linear dependency between the feature variable and the target. This means the score might underestimate the relation between a feature and the target if the relationship is nonlinear.
- The mutual information score is nonparametric and so can capture nonlinear relationships.

With `SelectKBest`, we define the number of features to keep, based on the score from the scoring function. Using `.fit_transform` we get back an array with only the selected features.

```python
# split data
train, valid, test = get_data_splits(baseline_data)
```

```python
# choose features
feature_cols = baseline_data.columns.drop("outcome")

# seperate predictors (X) and target (y)
X, y = train[feature_cols], train["outcome"]
```

```python
# keep 5-best features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=5)

X_new = selector.fit_transform(X, y)
```

Now we have our selected features. To drop the rejected features, we need to figure out which columns in the dataset were kept with `SelectKBest`. To do this, we can use `.inverse_transform` to get back an array with the shape of the original data. This returns a DataFrame with the same index and columns as the training set, but all the dropped columns are filled with zeros. We can find the selected columns by choosing features where the variance is non-zero.

```python
# get back the features we've kept
selected_features = pd.DataFrame(
    selector.inverse_transform(X_new), index=X.index, columns=feature_cols
)

# drop columns have values of all 0s
selected_columns = selected_features.columns[selected_features.var() != 0]
```

With this method we can choose the best K features, but we still have to choose K ourselves. How would we find the "best" value of K?

- To find the best value of K, we can fit multiple models with increasing values of K, then choose the smallest K with validation score above some threshold or some other criteria. That is, we're keeping the best features w/o degrading the model's performance.

### L1 Regularization

Univariate methods consider only one feature at a time when making a selection decision. Instead, we can make our selection using all of the features by including them in a linear model with L1 (Lasso) regularization which penalizes the absolute magnitude of the coefficients, or L2 (Ridge) regression which penalizes the square of the coefficients.

In general, feature selection with L1 regularization is more powerful the univariate tests, but it can also be very slow when you have a lot of data and a lot of features. As the strength of regularization is increased, features which are less important for predicting the target are set to 0.

From `sklearn.linear_model` module, you can use `Lasso` for regression problems, or `LogisticRegression` for classification. These can be used along with `sklearn.feature_selection.SelectFromModel` to select the non-zero coefficients. Otherwise, the code is similar to the univariate tests.

```python
# split data
train, valid, test = get_data_splits(baseline_data)
```

```python
# choose features
feature_cols = baseline_data.columns.drop("outcome")

# seperate predictors (X) and target (y)
X, y = train[feature_cols], train["outcome"]
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(X, y)
selector = SelectFromModel(logistic, prefit=True)

X_new = selector.transform(X)
```

- `C` is the inverse of regularization strength, which leads to some number of features being dropped. However, by setting C we aren't able to choose a certain number of features to keep. To find the regularization parameter that leaves the desired number of features, we can iterate over models with different regularization parameters from low to high and choose the one that leaves K features.
- `prefit` determines a model to be passed into the constructor directly or not. If True, `transform` must be called directly and `SelectFromModel` cannot be used with `cross_val_score`, `GridSearchCV` and similar utilities that clone the estimator. Otherwise train the model using `fit` and then `transform` to do feature selection.

```python
# get back the features we've kept
selected_features = pd.DataFrame(
    selector.inverse_transform(X_new), index=X.index, columns=feature_cols
)

# drop columns have values of all 0s
selected_columns = selected_features.columns[selected_features.var() != 0]
```

### Feature Selection with Trees

Since we're using a tree-based model, using another tree-based model for feature selection might produce better results. We could use something like `RandomForestClassifier` or `ExtraTreesClassifier` to find feature importances. `SelectFromModel` can use the feature importances to find the best features.

# Deep Learning
*Use TensorFlow to take machine learning to the next level. Your new skills will amaze you.*

## Intro to DL for Computer Vision
*A quick overview of how models work on images. [#](https://www.kaggle.com/dansbecker/intro-to-dl-for-computer-vision)*

### Intro

Deep Learning has revolutionized computer vision and it is the core technology behind capabilities like self driving cars.

For this series, we will use tensorflow and the tensorflow implementation of keras. Tensorflow is the most popular tool for deep learning and keras is a popular api or for specifying deep learning models. This allows you to specify models with the elegant of keras while taking greater advantage of some powerful tensorflow features.

Your machine learning experience so far has probably focused on tabular data like you would work with in pandas. We will be doing deep learning with images so we will start an overview of how images are stored for machine learning.

Images are composed of pixels.

- In b/w images, some pixels are completely black, some are completely white, and some are varying shades of grey. The pixels are arranged in rows and columns so it is natural to store them in a matrix. Each value represents the darkness of the pixel in the image
- Colour images have an extra dimension. For each pixel we store how red that pixel is, how green it is, and how blue it is. It's like a stack of three matrices.

**Tensor** is the word for something that is like a matrix but which can have any number of dimensions.

Today's deep learning models apply something called **convolutions** to this type of tensor. A convolution is a small tensor that can be multiplied over little sections of the main image. Convolutions are also called filters because depending on the values in that convolutional array, you can pick out the percific **patterns** from the image, like horizontal or vertical lines. The numerical values in the convolution determine what pattern it detects.

![](img/tensor.png)

You don't directly choose the numbers to go into your convolutions for deep learning... instead the deep learning technique determines what convolutions will be useful from the data (as part of model-training). But looking closely at convolutions and how they are applied to your image will improve your intuition for these models, how they work, and how to debug them when they don't work.

For example, a convolution that detected horizontal lines:

```python
horizontal_line_conv = [[1, 1], 
                        [-1, -1]]
```

or, a vertical line detector:

```python
vertical_line_conv = [[1, -1],
                      [1, -1]]
```

While any one convolution measures only a single pattern, there are more possible convolutions that can be created with large sizes. So there are also more patterns that can be captured with large convolutions. Does this mean powerful models require extremely large convolutions? Not necessarily.

## Building Models From Convolutions
*Scale up from simple building blocks to models with beyond human capabilities. [#](https://www.kaggle.com/dansbecker/building-models-from-convolutions)*

## TensorFlow Programming
*Start writing code using TensorFlow and Keras. [#](https://www.kaggle.com/dansbecker/tensorflow-programming)*

## Transfer Learning
*A powerful technique to build highly accurate models even with limited data. [#](https://www.kaggle.com/dansbecker/transfer-learning)*

## Data Augmentation
*Learn a simple trick that effectively increases amount of data available for model training. [#](https://www.kaggle.com/dansbecker/data-augmentation)*

## A Deeper Understanding of Deep Learning
*How Stochastic Gradient Descent and Back-Propagation train your deep learning model. [#](https://www.kaggle.com/dansbecker/a-deeper-understanding-of-deep-learning)*

## Deep Learning From Scratch
*Build models without transfer learning. Especially important for uncommon image types. [#](https://www.kaggle.com/dansbecker/deep-learning-from-scratch)*

## Dropout and Strides for Larger Models
*Make your models faster and reduce overfitting. [#](https://www.kaggle.com/dansbecker/dropout-and-strides-for-larger-models)*

# Intro to SQL
*Learn SQL for working with databases, using Google BigQuery to scale to massive datasets.*

# Advanced SQL
*Take your SQL skills to the next level.*

# Microchallenges
*Solve ultra-short challenges to build and test your skill.*

# Machine Learning Explainability
*Extract human-understandable insights from any machine learning model.*

# Natural Language Processing
*Distinguish yourself by learning to work with text data.*
