# Data Scientist's Roadmap
A minimalist roadmap to The Data Science World, based on [Kaggle's Roadmap](https://www.kaggle.com/learn/overview)

## **Python**
Learn the most important language for data science.

### [Hello, Python](https://www.kaggle.com/colinmorris/hello-python)
A quick introduction to Python syntax, variable assignment, and numbers

- Variable Assignment
- Function Calls
- Numbers and Arithmetic in Python
  ```python
  /     # true division
  //    # floor division
  %     # modulus
  **    # exponentiation
  ```
- Order of Operators: **PEMDAS**
- Builtin Functions for Working with Numbers
  ```python
  min()
  max()
  abs()
  # conversion functions
  int()
  float()
  ```

### [Functions and Getting Help](https://www.kaggle.com/colinmorris/functions-and-getting-help)
Calling functions and defining our own, and using Python's builtin documentation

- Getting Help
  - on modules, objects, instances, and ...
    ```python
    help()
    dir()
    ```
- Functions
  - `def func_name(vars):`
  - Docstrings, that `help()` returns
    ```python
    """ some useful info about the function """
    ```
  - Functions that don't Return
  - Default Arguments
    ```python
    print(..., sep='\t')
    ```
  - Functions Applied to Functions
    ```python
    fn(fn(arg))
    string.lower().split()
    ```

### [Booleans and Conditionals](https://www.kaggle.com/colinmorris/booleans-and-conditionals)
Using booleans for branching logic

- Booleans
  - `True` or `False` or `bool()`
  - Comparison Operations
    ```python
    a == b    # a equal to b
    a != b    # a not equal to b
    a <  b    # a less than b
    a >  b    # a greater than b
    a <= b    # a less than or equal to b
    a >= b    # a greater than or equal to b
    ```
  - Order of Operators: **PEMDAS** combined with Boolean Values
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
- Conditionals
  ```python
  if
  elif
  else
  ```
- Trues and Falses
  - `bool()`
  - All numbers are treated as **true**, except `0`.
  - All strings are treated as **true**, except the empty string `""`.
  - Empty sequences (strings, lists, tuples, sets)  are falsey and the rest are **truthy**.
- Conditional Expressions
  - Setting a variable to either of two values depending on a condition
    ```python
    outcome = 'failed' if grade < 50 else 'passed'
    ```

### [Lists](https://www.kaggle.com/colinmorris/lists)
Lists and the things you can do with them. Includes indexing, slicing and mutating

- Lists
  - `[]` or `list()`
  - **Mutable**
  - A mix of same or different types of variables
  - Indexing
    ```python
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    # first element
    planets[0]
    # last element
    planets[-1]
    ```
  - Slicing
    ```python
    planets[:3]
    planets[-3:]
    ```
  - Changing Lists
    ```python
    planets[:3] = ['Mur', 'Vee', 'Ur']
    ```
  - List Functions
    ```python
    len()
    sorted()
    max()
    sum()
    any()
    ```
  - Python Attributes & Methods (**Everything is an Object**)
    ```python
    # complex number object
    c = 12 + 5j
    c.imag
    c.real
    # integer number object
    x = 12
    x.bit_length()
    ```
  - List Methods
    ```python
    list.append()
    list.pop()
    list.index()
    in
    ```
- Tuples
  - `()` or `, , ,` or `tuple()`
  - **Immutable**
    ```python
    x = 0.125
    numerator, denominator = x.as_integer_ratio()
    ```

### [Loops and List Comprehensions](https://www.kaggle.com/colinmorris/loops-and-list-comprehensions)
For and while loops, and a much-loved Python feature: list comprehensions 

- Loops
  - Use in every iteratable objects: list, tuples, strings, ...
  - `for - in - :`
  - `range()`
  - `while` loops
- List Comprehensions
  - `[- for - in -]`
    ```python
    squares = [n**2 for n in range(10)]
    # constant
    [32 for planet in planets]
    # with if
    short_planets = [planet.upper() + "!" for planet in planets if len(planet) < 6]
    # combined with other functions
    return len([num for num in nums if num < 0])
    return sum([num < 0 for num in nums])
    return any([num % 7 == 0 for num in nums])
    ```
  - Solving a problem with less code is always nice, but it's worth keeping in mind the following lines from **The Zen of Python**.
    > Readability counts.<br>
    > Explicit is better than implicit.
- Enumerate
  - `for index, item in enumerate(items):`

### [String and Directories](https://www.kaggle.com/colinmorris/strings-and-dictionaries)
Working with strings and dictionaries, two fundamental Python data types 

- Strings
  - `''` or `""` or `""" """` or `str()`
  - Escaping with `\`
  - List-like
    ```python
    [char + '! ' for char in "Planet"]
    >>> ['P! ', 'l! ', 'a! ', 'n! ', 'e! ', 't! ']
    ```
  - **Immutable**
    ```python
    "Planet"[0] = 'M'
    >>> TypeError: 'str' object does not support item assignment
    ```
  - String Methods
    ```python
    str.upper()
    str.lower()
    str.index()
    str.startswith()
    str.endswith()
    ```
  - String and List, Back and Forward
    ```python
    # split
    year, month, day = '2020-03-05'.split('-')
    year, month, day
    >>> ('2020', '03', '05')
    # join
    '/'.join([month, day, year])
    >>> '03/05/2020'
    ```
  - String Formatting
    ```python
    "{}".format()
    f"{}"
    ```
- Dictionaries
  - `{}` or `dict()`
  - Pairs of keys,values
    ```python
    numbers = {'one':1, 'two':2, 'three':3}
    numbers['one']
    numbers['eleven'] = 11
    ```
  - Dictionary Comprehensions
    ```python
    planet_to_initial = {planet: planet[0] for planet in planets}
    ```
  - Access to all Keys or all Values
      ```python
      dict.keys()
      dict.values()
      ' '.join(sorted(planet_to_initial.values()))
      ```
  - Get key by value
    ```python
    key_of_min_value = min(numbers, key=numbers.get)
    ```
  - `in`
  - Loops in Dictionaries
    - A for loop over a dictionary will loop over its Keys
    - For loop over (key, value) pairs, use `item`
      ```python
      for planet, initial in planet_to_initial.items():
          print(f"{planet} begins with \"{initial}\"")
      ```

### [Working with External Libraries](https://www.kaggle.com/colinmorris/working-with-external-libraries)
Imports, operator overloading, and survival tips for venturing into the world of external libraries

- Imports
  - Simple import, `.` access
    ```python
    import math
    math.pi
    ```
  - `as` import, short `.` access
    ```python
    import math as mt
    mt.pi
    ```
  - `*` import, simple access
    ```python
    from math import *
    pi
    ```
    > The problem of * import is that some modules (ex. `math` and `numpy`) have functions with same name (ex. `log`) but with different semantics. So one of them overwrites (or "shadows") the other. It is called **overloading**.
  - Combined, solution for the `*` import
    ```python
    from math import log, pi
    from numpy import asarray
    ```
- Submodules
  - Modules contain variables which can refer to functions or values. Sometimes they can also have variables referring to other modules.
    ```python
    import numpy
    dir(numpy.random)
    >>> ['set_state', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'test', 'triangular', 'uniform', ...]
    # make an array of random numbers
    rolls = numpy.random.randint(low=1, high=6, size=10)
    rolls
    >>> array([3, 2, 5, 2, 4, 2, 2, 3, 2, 3])
    ```
  - Get Help
    - Standard Python datatypes are: **int**, **float**, **bool**, **list**, **string**, and **dict**.
    - As you work with various libraries for specialized tasks, you'll find that they define their own types. For example
      - Matplotlib: **Subplot**, **Figure**, **TickMark**, and **Annotation**
      - Pandas: **DataFrame** and **Serie**
      - Tensorflow: **Tensor**
    - Use `type()` to find the type of an object.
    - Use `dir()` and `help()` for more details.
      ```python
      dir(umpy.ndarray)
      >>> [...,'__bool__', ..., '__delattr__', '__delitem__', '__dir__', ..., '__sizeof__', ..., 'max', 'mean', 'min', ..., 'sort', ..., 'sum', ..., 'tobytes', 'tofile', 'tolist', 'tostring', ...]
      ```
- Operator Overloading
  - Index
    ```python
    # list
    xlist = [[1,2,3], [2,4,6]]
    xlist[1,-1]
    >>> TypeError: list indices must be integers or slices, not tuple
    # numpy array
    xarray = numpy.asarray(xlist)
    xarray[1,-1]
    >>> 6
    ```
  - Add
    ```python
    # list
    [3, 4, 1, 2, 2, 1] + 10
    >>> TypeError: can only concatenate list (not "int") to list
    # numpy array
    rolls + 10
    >>> array([13, 12, 15, 12, 14, 12, 12, 13, 12, 13])
    # tensorflow
    import tensorflow as tf
    a = tf.constant(1)
    b = tf.constant(1)
    a + b
    >>> <tf.Tensor 'add:0' shape=() dtype=int32>
    ```
  - When Python programmers want to define how operators behave on their types, they do so by implementing **Dunder/Special Methods**, methods with special names beginning and ending with 2 underscores such as `__add__` or `__contains__`.
  - More info: https://is.gd/3zuhhL

## **Intro to Machine Learning**
Learn the core ideas in machine learning, and build your first models.

### [How Models Work](https://www.kaggle.com/dansbecker/how-models-work)
The first step if you're new to machine learning

- Introduction
  > You ask your cousin how he's predicted real estate values in the past. and he says it is just intuition. But more questioning reveals that he's identified price patterns from houses he has seen in the past, and he uses those patterns to make predictions for new houses he is considering. Machine learning works the same way.
  - **Fitting** or **Training**: Capturing patterns from **training data**
  - **Predicting**: Getting results from applying the model to **new data**
- Decision Tree

### [Basic Data Exploration](https://www.kaggle.com/dansbecker/basic-data-exploration)
Load and understand your data

- Using Pandas to Get Familiar with the Data
  - **DataFrame**: The most important part of the Pandas library, similar to a sheet in Excel, or a table in a SQL database
    ```python
    # read data from csv file and store it in pandas DataFrame
    import pandas as pd
    melbourne_data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
    
    # statistical summary
    melbourne_data.describe()
    ```
    |       |  Price  | Rooms | Bedroom2 | Bathroom | Landsize | BuildingArea | YearBuilt | Lattitude | Longtitude |  ...  |
    | :---: | :-----: | :---: | :------: | :------: | :------: | :----------: | :-------: | :-------: | :--------: | :---: |
    | count |  13580  | 13580 |  13580   |  13580   |  13580   |     7130     |   8205    |   13580   |   13580    |  ...  |
    | mean  | 1075684 | 2.93  |   2.91   |   1.53   |  558.41  |    151.96    |  1964.68  |  -37.80   |   144.99   |  ...  |
    |  std  | 639310  | 0.95  |   0.96   |   0.69   | 3990.66  |    541.01    |   37.27   |   0.07    |    0.10    |  ...  |
    |  min  |  85000  |   1   |    0     |    0     |    0     |      0       |   1196    |  -38.18   |   144.43   |  ...  |
    |  25%  | 650000  |   2   |    2     |    1     |   177    |      93      |   1940    |  -37.85   |   144.92   |  ...  |
    |  50%  | 903000  |   3   |    3     |    1     |   440    |     126      |   1970    |  -37.80   |   145.00   |  ...  |
    |  75%  | 1330000 |   3   |    3     |    2     |   651    |     174      |   1999    |  -37.75   |   145.05   |  ...  |
    |  max  | 9000000 |  10   |    20    |    8     |  433014  |    44515     |   2018    |  -37.40   |   145.52   |  ...  |
  - Interpreting Data Description
    - `count`: shows how many rows have non-missing values.
    - `mean`: the average.
    - `std`: the standard deviation, measures how numerically spread out the values are.
    - `min`, `25%` (25th percentile), `50%` (50th percentiles), `75%` (75th percentiles) and `max`

### [Your First Machine Learning Model](https://www.kaggle.com/dansbecker/your-first-machine-learning-model)
Building your first model. Hurray!

- Selecting Data for Modeling
  - Datasets have too many variables to wrap your head around. We'll start by picking a few variables using our intuition. Later, we use statistical techniques to automatically prioritize variables.
    ```python
    # look at the list of all columns in the dataset
    melbourne_data.columns
    
    # filter rows with missing values
    dropna_melbourne_data = melbourne_data.dropna(axis=0)
    ```
  - Selecting the **Prediction Target** (`y`)
    ```python
    y = dropna_melbourne_data["Price"]
    ```
  - Choosing **Features** (input columns, `X`)
    ```python
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
    
    # quick look at the data we'll be using to predict house prices
    X.describe()
    X.head()
    ```
- Building the Model
  - Steps
    - **Define**: What type of model will it be? A decision tree? Some other type of model?
    - **Fit**: Capture patterns from provided data. This is the heart of modeling.
    - **Predict**: Just what it sounds like.
    - **Evaluate**: Determine how accurate the model's predictions are.
  - scikit-learn
    ```python
    # define model
    # in random processes like defining a model, `random_state` ensures you get the same results in each run
    from sklearn.tree import DecisionTreeRegressor
    melbourne_model = DecisionTreeRegressor(random_state=1)
    
    # fit model
    melbourne_model.fit(X, y)
    
    # make prediction
    predictions = melbourne_model.predict(X)
    ```

### [Model Validation](https://www.kaggle.com/dansbecker/model-validation)
Measure the performance of your model ? so you can test and compare alternatives

- Summarizing the Model Quality into Metrics
  - There are many metrics for summarizing the model quality.
  - **Predictive Accuracy**: Will the model's predictions be close to what actually happens?
    - **Mean Absolute Error** (MAE)
      ```python
      from sklearn.metrics import mean_absolute_error
      mean_absolute_error(y, predictions)
      ```
- **Big Mistake**: Measuring scores with the training data or the problem with **"In-Sample" scores**!
- **Validation Data**
  - **Making Predictions on New Data**
  - The most straightforward way to do that is to exclude some data from the model-building process, and then use those to test the model's accuracy.
    ```python
    # break off validation set from training data, for both features and target
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
    
    # define model
    melbourne_model = DecisionTreeRegressor(random_state=1)
    
    # fit model
    melbourne_model.fit(X_train, y_train)
    
    # make prediction on validation data
    predictions_val = melbourne_model.predict(X_valid)
    
    # evaluate the model
    mean_absolute_error(y_valid, predictions_val)
    ```
    > The MAE for the in-sample data was about 500 dollars. For out-of-sample data, it's more than 250,000 dollars. As a point of reference, the average home value in the validation data is 1.1 million dollars. So the error in new data is about a quarter of the average home value.
- There are many ways to improve a model, such as
  - Finding **better features**, the iterating process of building models with different features and comparing them to each other
  - Finding **better model types**

### [Underfitting and Overfitting](https://www.kaggle.com/dansbecker/underfitting-and-overfitting)
Fine-tune your model for better performance.

- Experimenting with Different Models
  - **Over-fitting**: Capturing spurious patterns that won't recur in the future, leading to less accurate predictions.
  - **Under-fitting**: Failing to capture relevant patterns, again leading to less accurate predictions.
  - In the **Decision Tree** model, the most important option to control the accuracy is the **tree's depth**, a measure of how many splits it makes before coming to a prediction.
    - A **deep tree** makes leaves with fewer objects. It causes **over-fitting**.
    - A **shallow tree** makes big groups. It causes **under-fitting**.
    - There are a few options for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes. But the `max_leaf_nodes` argument provides a very sensible way to control overfitting vs underfitting.
      ```python
      # function for comparing MAE with differing values of max_leaf_nodes
      def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):
          model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
          model.fit(X_train, y_train)
          predictions_val = model.predict(X_valid)
          mae = mean_absolute_error(y_valid, predictions_val)
          return(mae)
      
      # compare models
      max_leaf_nodes_candidates = [5, 50, 500, 5000]
      scores = {
          leaf_size: get_mae(leaf_size, X_train, X_valid, y_train, y_valid)
          for leaf_size in max_leaf_nodes_candidates
      }
      best_tree_size = min(scores, key=scores.get)
      ```

### [Random Forests](https://www.kaggle.com/dansbecker/random-forests)
Using a more sophisticated machine learning algorithm.

- Introduction
  - Decision trees leave you with a difficult decision. A deep tree and over-fitting vs. a shallow one and under-fitting. Even today's most sophisticated modeling techniques face this tension. But, many models have clever ideas that can lead to better performance.
- **Random Forest**
  - A Random Forest model uses many trees, and makes a prediction by averaging the predictions of each component. It generally has much better predictive accuracy even with than a single decision tree, even with default parameters, without tuning the parameters like `max_leaf_nodes`.
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
- Some models, like the **XGBoost** model, provides better performance when tuned well with the right parameters (but which requires some skill to get the right model parameters).

### [Exercise: Machine Learning Competitions](https://www.kaggle.com/kernels/fork/1259198)
Enter the world of machine learning competitions to keep improving and see your progress

- Setup
  ```python
  # load data
  import pandas as pd
  X_full = pd.read_csv("../input/train.csv", index_col="Id")
  X_test_full = pd.read_csv("../input/test.csv", index_col="Id")
  
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

  # break off validation set from training data
  from sklearn.model_selection import train_test_split
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
  ```
- Evaluate Several Models
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
  
  # function for comparing different models
  from sklearn.metrics import mean_absolute_error
  def score_model(model, X_train, X_valid, y_train, y_valid):
      # fit model
      model.fit(X_train, y_train)
      # make validation predictions
      preds_valid = model.predict(X_valid)
      # return mae
      return mean_absolute_error(y_valid, preds_valid)
  
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
- Generate Test Predictions
  ```python
  # define model, based on the most accurate model
  my_model = RandomForestRegressor(n_estimators=100, criterion="mae", random_state=0)

  # fit the model to the training data, all of it
  my_model.fit(X, y)

  # make test prediction
  preds_test = my_model.predict(X_test)

  # save predictions in format used for competition scoring
  output = pd.DataFrame({"Id": X_test.index, "SalePrice": preds_test})
  output.to_csv("submission.csv", index=False)
  ```

## **Intermediate Machine Learning**
Learn to handle missing values, non-numeric values, data leakage and more. Your models will be more accurate and useful.

### [Introduction](https://www.kaggle.com/alexisbcook/introduction)
Review what you need for this Micro-Course

In this micro-course, you will accelerate your machine learning expertise by learning how to:
* Tackle data types often found in real-world datasets (**missing values**, **categorical variables**),
* Design **pipelines** to improve the quality of your machine learning code,
* Use advanced techniques for model validation (**cross-validation**),
* Build state-of-the-art models that are widely used to win Kaggle competitions (**XGBoost**), and
* Avoid common and important data science mistakes (**leakage**).

### [Missing Values](https://www.kaggle.com/alexisbcook/missing-values)
Missing values happen. Be prepared for this common challenge in real datasets.

- Introduction
  - There are many ways data can end up with missing values. For example,
    - A 2 bedroom house won't include a value for the size of a third bedroom.
    - A survey respondent may choose not to share his income.
  - Most machine learning libraries (including scikit-learn) give an error if you try to build a model using data with missing values.
  - To show number of missing values in each column
    ```python
    def missing_val_count(data):
        missing_val_count_by_column = data.isnull().sum()
        return missing_val_count_by_column[missing_val_count_by_column > 0]
    ```
- Approaches
  - Setup
    ```python
    # load data
    import pandas as pd
    X_full = pd.read_csv("../input/train.csv", index_col="Id")
    X_test_full = pd.read_csv("../input/test.csv", index_col="Id")

    # remove rows with missing "SalePrice"
    X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)

    # separate target (y) from features (X)
    y = X_full["SalePrice"]
    X_full.drop(["SalePrice"], axis=1, inplace=True)

    # use only numerical features, to keep things simple
    X = X_full.select_dtypes(exclude=["object"])
    X_test = X_test_full.select_dtypes(exclude=["object"])

    # break off validation set from training data
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )

    # get names of columns with missing values
    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    # function for comparing different approaches
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        preds_valid = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds_valid)
    ```
  - A Simple Option: **Drop** Columns with Missing Values
    - The model loses access to a lot of (potentially useful!) information with this approach.
      ```python
      # drop `cols_with_missing` in training and validation data
      reduced_X_train = X_train.drop(cols_with_missing, axis=1)
      reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
      
      # evaluate the model
      score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)
      ```
  - A Better Option: **Imputation**
    - Imputation fills in the missing values with some number.
    - Strategy:
      - default=`mean` replaces missing values using the mean along each column. (only numeric)
      - `median` replaces missing values using the median along each column. (only numeric)
      - `most_frequent` replaces missing using the most frequent value along each column. (strings or numeric)
      - `constant` replaces missing values with `fill_value`. (strings or numeric)
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
- Train and Evaluate Model
  ```python
  # define and fit model
  model = RandomForestRegressor(n_estimators=100, random_state=0)
  model.fit(imputed_X_train, y_train)
  
  # make validation prediction
  preds_valid = model.predict(imputed_X_valid)
  mean_absolute_error(y_valid, preds_valid)
  ```
- Test Data
  ```python
  # preprocess test data
  imputed_X_test = pd.DataFrame(imputer.fit_transform(X_test))
  # put column names back
  imputed_X_test.columns = X_test.columns
  
  # make test prediction
  preds_test = model.predict(imputed_X_test)
  
  # save test predictions to file
  output = pd.DataFrame({"Id": X_test.index, "SalePrice": preds_test})
  output.to_csv("submission.csv", index=False)
  ```

### [Categorical Variables](https://www.kaggle.com/alexisbcook/categorical-variables)
There's a lot of non-numeric data out there. Here's how to use it for machine learning

- Introduction
  - A categorical variable takes only a limited number of values.
    - Ordinal: A question that asks "how often you eat breakfast?" and provides four options: "Never", "Rarely", "Most days", or "Every day".
    - Nominal: A question that asks "what brand of car you own?".
  - Most machine learning libraries (including scikit-learn) give an error if you try to build a model using data with categorical variables.
- Approaches
  - Setup
    ```python
    # load data
    import pandas as pd
    X_full = pd.read_csv("../input/train.csv", index_col="Id")
    X_test_full = pd.read_csv("../input/test.csv", index_col="Id")

    # remove rows with missing target
    X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)

    # separate target (y) from features (X)
    y = data["Price"]
    X = data.drop(["Price"], axis=1)

    # break off validation set from training data
    from sklearn.model_selection import train_test_split
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )

    # handle missing values (simplest approach)    
    cols_with_missing = [
        col for col in X_train_full.columns if X_train_full[col].isnull().any()
    ]
    X_train_full.drop(cols_with_missing, axis=1, inplace=True)
    X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

    # select categorical columns with relatively low cardinality, to keep things simple
    # cardinality means the number of unique values in a column
    categorical_cols = [
        cname
        for cname in X_train_full.columns
        if (X_train_full[cname].dtype == "object") and (X_train_full[cname].nunique() < 10)
    ]

    # select numerical columns
    numerical_cols = [
        cname
        for cname in X_train_full.columns
        if X_train_full[cname].dtype in ["int64", "float64"]
    ]

    # keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # function for comparing different approaches
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    def score_dataset(X_train, X_valid, y_train, y_valid):
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)
    ```
  - **Drop** Categorical Variables
    - This approach will only work well if the columns did not contain useful information.
      ```python
      # drop catagorial columns
      drop_X_train = X_train.select_dtypes(exclude=["object"])
      drop_X_valid = X_valid.select_dtypes(exclude=["object"])
      
      # evaluate the model
      score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
      ```
  - **Label Encoding**
    - Label encoding assigns each unique value, that appears in the training data, to a different integer.
    - In the case that the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them.
    - It should be used only for target labels encoding.
    - To encode categorical features, use One-Hot Encoder, which can handle unseen values.
    - For **tree-based models** (like decision trees and random forests), you can expect label encoding to work well with **ordinal** variables.
      ```python
      # find columns, which are in validation data but not in training data
      good_label_cols = [
          col for col in categorical_cols if set(X_train[col]) == set(X_valid[col])
      ]
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
  - **One-Hot Encoding**
    - One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data. Useful parameters are:
      - `handle_unknown="ignore"` avoids errors when the validation data contains classes that aren't represented in the training data,
      - `sparse=False` returns the encoded columns as a numpy array (instead of a sparse matrix).
    - In contrast to label encoding, one-hot encoding does not assume an ordering of the categories. Thus, you can expect this approach to work particularly well with categorical variables without an intrinsic ranking, we refer them as **nominal** variables.
    - One-hot encoding generally does **not** perform well with high-cardinality categorical variable (i.e., more than 15 different values). **Cardinality** means the number of unique values in a column.
      ```python
      # get cardinality for each column with categorical data
      object_nunique = list(map(lambda col: X_train[col].nunique(), categorical_cols))
      d = dict(zip(categorical_cols, object_nunique))

      # print cardinality by column, in ascending order
      sorted(d.items(), key=lambda x: x[1])
      ```
    - For this reason, we typically will only one-hot encode columns with relatively low cardinality. Then, high cardinality columns can either be dropped from the dataset, or we can use label encoding.
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
- Doing all things seperately for training, evaluating and testing is way DIFFICULT. Doing with Pipelines is **FUN**!

### [Pipelines](https://www.kaggle.com/alexisbcook/pipelines)
A critical skill for deploying (and even testing) complex models with pre-processing

- Introduction
  - Pipelines are a simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline **bundles preprocessing and modeling steps** so you can use the whole bundle as if it were a single step.
  - Some important benefits of pipelines are:
    - **Cleaner Code**: Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step.
    - **Fewer Bugs**: There are fewer opportunities to misapply a step or forget a preprocessing step.
    - **Easier to Productionize**: It can be surprisingly hard to transition a model from a prototype to something deployable at scale, but pipelines can help.
    - **More Options for Model Validation**: You will see an example in the Cross-Validation tutorial.
- Steps
  - Setup
    ```python
    # load data
    import pandas as pd
    X_full = pd.read_csv("../input/train.csv", index_col="Id")
    X_test_full = pd.read_csv("../input/test.csv", index_col="Id")

    # remove rows with missing target
    X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)

    # separate target (y) from features (X)
    y = X_full["Price"]
    X = X_full.drop(["Price"], axis=1)

    # break off validation set from training data
    from sklearn.model_selection import train_test_split
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )

    # select categorical columns with relatively low cardinality, to keep things simple
    # cardinality means the number of unique values in a column
    categorical_cols = [
        cname
        for cname in X_train_full.columns
        if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
    ]

    # select numerical columns
    numerical_cols = [
        cname
        for cname in X_train_full.columns
        if X_train_full[cname].dtype in ["int64", "float64"]
    ]

    # keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()
    ```
  - Step 1: **Define Preprocessing Steps**
    - Similar to how a pipeline bundles together preprocessing and modeling steps, we use the `ColumnTransformer` class to bundle together different preprocessing steps. The code below:
      - imputes missing values in numerical data, and
      - imputes missing values and applies a one-hot encoding to categorical data.
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
  - Step 2: **Define the Model**
    - For example we use `RandomForestRegressor`
      ```python
      from sklearn.ensemble import RandomForestRegressor
      model = RandomForestRegressor(n_estimators=100, min_samples_split=3, random_state=0)
      ```
  - Step 3: **Create and Evaluate the Pipeline**
    - We use the `Pipeline` class to define a pipeline that bundles the preprocessing and modeling steps.
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
- Test
  ```python
  # preprocessing of test data, fit model
  preds_test = my_pipeline.predict(X_test)

  # save predictions in format used for competition scoring
  output = pd.DataFrame({"Id": X_test.index, "SalePrice": preds_test})
  output.to_csv("submission.csv", index=False)
  ```
- **Vola**!

### [Cross-Validation](https://www.kaggle.com/alexisbcook/cross-validation)
A better way to test your models

### [XGBoost](https://www.kaggle.com/alexisbcook/xgboost)
The most accurate modeling technique for structured data

### [Data Leakage](https://www.kaggle.com/alexisbcook/data-leakage)
Find and fix this problem that ruins your model in subtle ways

## **Data Visualization**

Make great data visualizations. A great way to see the power of coding!

## **Pandas**

Solve short hands-on challenges to perfect your data manipulation skills.

## **Feature Engineering**

Discover the most effective way to improve your models.

## **Deep Learning**

Use TensorFlow to take machine learning to the next level. Your new skills will amaze you.

## **Intro to SQL**

Learn SQL for working with databases, using Google BigQuery to scale to massive datasets.

## **Advanced SQL**

Take your SQL skills to the next level.

## **Geopatial Analysis**

Create interactive maps, and discover patterns in geospatial data.

## **Microchallenges**

Solve ultra-short challenges to build and test your skill.

## **Machine Learning Explainability**

Extract human-understandable insights from any machine learning model.

## **Natural Language Processing**

Distinguish yourself by learning to work with text data.
