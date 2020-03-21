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

- Introduction
  - Machine learning is an **iterative process**. You will face choices about what **predictive variables** to use, what **types of models** to use, what **arguments** to supply to those models, etc.
  - In a dataset with 5000 rows, you will typically keep about 20% of the data as a validation dataset, or 1000 rows. But this leaves some random chance in determining model scores. That is, a model might do well on one set of 1000 rows, even if it would be inaccurate on a different 1000 rows.
  - The larger the validation set, the less randomness (aka "noise") there is in our measure of model quality.
- Cross-Validation
  - In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality.
  - In Experiment 1, we use the first **fold** (20%) as a **validation (or holdout) set** and everything else as training data. We repeat this process, using every fold once as the holdout set.
  - Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset.
  - Cross-validation gives a more accurate measure of model quality. However, it can take longer to run.
  - For **small datasets**, you should run cross-validation. But for larger datasets, a single validation set is sufficient.
  - There's no simple threshold for what constitutes a large vs. small dataset. But if your model takes a couple minutes or less to run, it's probably worth switching to cross-validation. Or you can run cross-validation and see if the scores for each experiment seem close.
- Steps
  - Setup
    ```python
    # load data
    import pandas as pd
    X_full = pd.read_csv("../input/train.csv", index_col="Id")

    # remove rows with missing target
    X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)

    # separate target (y) from features (X)
    y = X_full["SalePrice"]
    X_full.drop(["SalePrice"], axis=1, inplace=True)

    # select numeric columns only
    numeric_cols = [
        cname for cname in X_full.columns if X_full[cname].dtype in ["int64", "float64"]
    ]
    X = X_full[numeric_cols].copy()
    X_test = test_data[numeric_cols].copy()
    ```
  - **Define a Pipeline**. It's difficult to do cross-validation without pipelines.
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
  - **Obtain the cross-validation scores** with the `cross_val_score()` function
    ```python
    from sklearn.model_selection import cross_val_score
    scores = -1 * cross_val_score(
        my_pipeline, X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    scores
    >>> [301628 303164 287298 236061 260383]
    
    # take the average score across experiments
    scores.mean()
    >>> 277707
    ```
    - The `cv` parameter sets the number of folds.
    - The `scoring` parameter chooses a measure of model quality to report. The docs for scikit-learn show a [list of options](http://scikit-learn.org/stable/modules/model_evaluation.html).
    - It is a little surprising that we specify **negative MAE**. Scikit-learn has a convention where all metrics are defined so a high number is better. Using negatives here allows them to be consistent with that convention. So multiply this score by -1.
  - **Combine** them as a function, for example on `n_estimators`, the number of trees in the random forest
    ```python
    def get_score(n_estimators):
        """return the average MAE over 3 CV folds of random forest model.
        
        Keyword argument:
        n_estimators -- the number of trees in the forest
        """
        
        my_pipeline = Pipeline(
            steps=[
                ("preprocessor", SimpleImputer()),
                ("model", RandomForestRegressor(n_estimators=n_estimators, random_state=0)),
            ]
        )

        scores = -1 * cross_val_score(
            my_pipeline, X, y, cv=3, scoring="neg_mean_absolute_error"
        )

        return scores.mean()
    ```
  - **Evalute** the model performance corresponding to some different values
    ```python
    # for example 50, 100, 150, ..., 300, 350, 400
    results = {i: get_score(i) for i in range(50, 450, 50)}
    ```
  - Find the best parameter value
    ```python
    import matplotlib.pyplot as plt
    %matplotlib inline
    plt.plot(results.keys(), results.values())
    plt.show()
    ```
  - If you'd like to learn more about **hyperparameter optimization**, you're encouraged to start with **grid search**, which is a straightforward method for determining the best combination of parameters for a machine learning model. Thankfully, scikit-learn also contains a built-in function `GridSearchCV()` that can make your grid search code very efficient!

### [XGBoost](https://www.kaggle.com/alexisbcook/xgboost)
The most accurate modeling technique for structured data

- Introduction
  - Ensemble methods combine the predictions of several models and achieve better performance, like Random Forest method.
- **Gradient Boosting**
  - It is a ensemble method that goes through cycles to iteratively add models into an ensemble.
  - Steps
    - It begins by initializing the ensemble with a **naive model**, even if its predictions are wildly inaccurate.
    - Then it uses the all models in the current ensemble to **generate predictions** for each observation in the dataset.
    - These predictions are used to **calculate a loss function** (like mean squared error, for instance).
    - Then, it uses the loss function to **fit a new model** that will be added to the ensemble and **reduce the loss**. (It uses [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) on the loss function to determine the parameters in this new model.)
    - Then it adds the new model to ensemble.
    - ... **repeat** steps 2-5!
- XGBoost
  - XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional features focused on performance and speed. (Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages.)
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
    y = X_full["SalePrice"]
    X = X_full.drop(["SalePrice"], axis=1, inplace=True)

    # break off validation set from training data
    from sklearn.model_selection import train_test_split
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )

    # select categorical columns with relatively low cardinality (convenient but arbitrary)
    # cardinality means the number of unique values in a column
    low_cardinality_cols = [
        cname
        for cname in X_train_full.columns
        if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
    ]

    # select numeric columns
    numeric_cols = [
        cname
        for cname in X_train_full.columns
        if X_train_full[cname].dtype in ["int64", "float64"]
    ]

    # keep selected columns only
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # one-hot encode the data
    # to shorten the code, we use pandas
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)
    X_train, X_valid = X_train.align(X_valid, join="left", axis=1)
    X_train, X_test = X_train.align(X_test, join="left", axis=1)
    ```
  - Use **`XGBRegressor`**, the scikit-learn API for XGBoost
    ```python
    # define model
    from xgboost import XGBRegressor
    my_model = XGBRegressor()
    
    # fit model
    my_model.fit(X_train, y_train)
    ```
  - Make Predictions
    ```python
    # make predictions
    predictions = my_model.predict(X_valid)
    
    # evaluate the model
    from sklearn.metrics import mean_absolute_error
    mean_absolute_error(predictions, y_valid)
    ```
  - **Parameter Tuning**
    ```python
    my_model = XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=4)
    my_model.fit(
        X_train, y_train,
        early_stopping_rounds=5,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    ```
    - `n_estimators`
      - It specifies how many times to go through the modeling cycle. It is equal to the number of models that we include in the ensemble.
      - **Too low** a value causes **underfitting**.
      - **Too high** a value causes **overfitting**.
      - Typical values range from 100-1000, though this depends a lot on the `learning_rate` parameter.
    - `early_stopping_rounds`
      - It offers a way to **automatically find the ideal value** for `n_estimators`.
      - Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for `n_estimators`.
      - Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping.
      - Setting `early_stopping_rounds=5` is a reasonable choice.
      - When using `early_stopping_rounds`, you also need to set aside some data for calculating the validation scores. This is done by setting the `eval_set` parameter.
    - `learning_rate`
      - Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in. This means each tree we add to the ensemble helps us less. So, we can set a **higher** value for `n_estimators` **without overfitting**.
      - If we use early stopping, the appropriate number of trees will be determined automatically.
      - In general, a **small learning rate** and **large number of estimators** will yield **more accurate XGBoost models**, though it will also take the model **longer to train** since it does more iterations through the cycle. As default, XGBoost sets `learning_rate=0.1`.
    - `n_jobs`
      - You can use parallelism to build your models faster. It's common to set the parameter `n_jobs` equal to the number of cores on your machine.
      - On smaller datasets, this won't help. But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.


### [Data Leakage](https://www.kaggle.com/alexisbcook/data-leakage)
Find and fix this problem that ruins your model in subtle ways

- Introduction
  - Data leakage (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction.
  - This causes a model to **look accurate** until you start making decisions with the model, and then the model becomes very inaccurate.
  - There are two main types of leakage: target leakage and train-test contamination.
- **Target Leakage**
  - It occurs when **your predictors include data becomes available after predictions**.
  - Example: People take antibiotic medicines after getting pneumonia in order to recover.
    - The data shows a strong relationship between those columns.
    - But `took_antibiotic_medicine` is frequently changed after the value for `got_pneumonia` is determined.
    - The model would see that anyone who has a value of `False` for `took_antibiotic_medicine` didn't have pneumonia.
    - Since validation data comes from the same source as training data, the pattern will repeat itself in validation, and the model will have great validation (or cross-validation) scores.
  - **Prevent**
    - Any variable updated (or created) after the target value is realized should be excluded.
- **Train-Test Contamination**
  - It occurs when **the validation data affects the preprocessing behavior**.
  - Example: Imagine you run preprocessing (like fitting an imputer for missing values) before calling `train_test_split()`.
  - This problem becomes even more dangerous when you do more complex feature engineering.
  - **Prevent**
    - If your validation is based on a simple train-test split, exclude the validation data from any type of fitting, including the fitting of preprocessing steps.
    - This is easier if you use scikit-learn **pipelines**.
    - When using cross-validation, it's even more critical that you do your preprocessing inside the pipeline!
- Examples
  - **The Data Science of Shoelaces**
    - Build a model to predict how many shoelaces NIKE needs each month.
    - The most important features in the model are
      - The current month
      - Advertising expenditures in the previous month
      - Various macroeconomic features (like the unemployment rate) as of the beginning of the current month
      - The amount of leather they ended up using in the current month
    - The results show the model is almost perfectly accurate if you include the feature about how much leather they used because the amount of leather they use is a **perfect indicator** of how many shoes they produce.
    - Is the leather used feature constitutes a source of data leakage?
    - **Solution**
      - It depends on details of **how data is collected** (which is common when thinking about leakage).
      - Would you at the beginning of the month decide how much leather will be used that month? If so, this is ok. But if that is determined during the month, you would not have access to it when you make the prediction.
    - You could use the amount of leather they ordered (rather than the amount they actually used) leading up to a given month as a predictor in your shoelace model.
    - Is this constitutes a source of data leakage?
    - **Solution**
      - This could be fine, but it depends on whether they order shoelaces first or leather first.
      - If they order shoelaces first, you won't know how much leather they've ordered when you predict their shoelace needs.
  - **Getting Rich with Cryptocurrencies**
    - Build a model to predict the price of a new cryptocurrency one day ahead.
    - The most important features in the model are
      - Current price of the currency
      - Amount of the currency sold in the last 24 hours
      - Change in the currency price in the last 24 hours
      - Change in the currency price in the last 1 hour
      - Number of new tweets in the last 24 hours that mention the currency
    - The value of the cryptocurrency in dollars has fluctuated up and down by over 100$ in the last year, and yet the model′s average error is less than 1$.
    - Do you invest based on this model?
    - **Solution**
      - There is no source of leakage here. These features should be available at the moment you want to make a predition, and they're unlikely to be changed in the training data after the prediction target is determined.
      - But, this model's accuracy could be misleading if you aren't careful.
      - If the price moves gradually, today's price will be an accurate predictor of tomorrow's price, but it may not tell you whether it's a good time to invest.
      - A better prediction target would be the change in price (up or down and by how much) over the next day.
  - **Housing Prices**
    - Build a model to predict housing prices.
    - The most important features in the model are
      - Size of the house (in square meters)
      - Average sales price of homes in the same neighborhood
      - Latitude and longitude of the house
      - Whether the house has a basement
    - Which of the features is most likely to be a source of leakage?
    - **Solution**:
      - Average sales price of homes in the same neighborhood is the source of target leakage.
        - We don't know the rules for when this is updated.
        - If the field is updated in the raw data after a home was sold, and the home's sale is used to calculate the average, this constitutes a case of target leakage.
        - At an extreme, if only one home is sold in the neighborhood, and it is the home we are trying to predict, then the average will be exactly equal to the value we are trying to predict.
        - In general, for neighborhoods with few sales, the model will perform very well on the training data. But when you apply the model, the home you are predicting won't have been sold yet, so this feature won't work the same as it did in the training data.
      - Other features don't change, and will be available at the time we want to make a prediction.
  - **Credit Card Applications**
    - Build a model to predict which applications were accepted.
      ```python
      # load data
      import pandas as pd
      data = pd.read_csv(
          "../input/aer-credit-card-data/AER_credit_card_data.csv",
          true_values=["yes"],
          false_values=["no"],
      )

      # separate target (y) from features (X)
      y = data["card"]
      X = data.drop(["card"], axis=1)
      ```
    - Since this is a small dataset, we will use cross-validation to ensure accurate measures of model quality.
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
    - With experience, you'll find that it's very rare to find models that are accurate 98% of the time. It happens, but it's uncommon enough that we should inspect the data more closely for **target leakage**.
    - Summary of the data
      ```
      card: 1 if credit card application accepted, 0 if not
      reports: Number of major derogatory reports
      age: Age n years plus twelfths of a year
      income: Yearly income (divided by 10,000)
      share: Ratio of monthly credit card expenditure to yearly income
      expenditure: Average monthly credit card expenditure
      owner: 1 if owns home, 0 if rents
      selfempl: 1 if self-employed, 0 if not
      dependents: 1 + number of dependents
      months: Months living at current address
      majorcards: Number of major credit cards held
      active: Number of active credit accounts
      ```
    - A few variables look suspicious. For example, does `expenditure` mean expenditure on this card or on cards used before appying?
      ```python
      # fraction of those who received a card and had no expenditures
      (X["expenditure"][y] == 0).mean()
      >>> 0.02

      # fraction of those who did not receive a card and had no expenditures
      (X["expenditure"][~y] == 0).mean()
      >>> 1.00
      ```
    - As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures. This seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.
    - Since `share` is partially determined by `expenditure`, it should be excluded too.
    - The variables `active` and `majorcards` are a little less clear, but from the description, they sound concerning.
      ```python
      # drop leaky features from dataset
      potential_leaks = ["expenditure", "share", "active", "majorcards"]
      X2 = X.drop(potential_leaks, axis=1)

      # evaluate the model, with leaky predictors removed
      cv_scores = cross_val_score(my_pipeline, X2, y, cv=5, scoring="accuracy")
      cv_scores.mean()
      >>> 0.831679
      ```

## **Pandas**
Solve short hands-on challenges to perfect your data manipulation skills.

### [Creating, Reading and Writing](https://www.kaggle.com/residentmario/creating-reading-and-writing)
You can't work with data if you can't read it. Get started here.

- Creating Data
  - DataFrame
    - It is a table.
    - It contains an array of individual entries, each of which has a certain value.
    - Each entry corresponds to a row (or record) and a column.
      ```python
      import pandas as pd
      pd.DataFrame({"Apples": [50, 21], "Bananas": [131, 2]})
      ```
    - The syntax for declaring a new one is a dictionary whose keys are the column names, and whose values are a list of entries.
    - The list of row labels used in a DataFrame is known as an Index. We can assign values to it by using an `index` parameter in our constructor
      ```python
      pd.DataFrame(
          {"Apples": [50, 21], "Bananas": [131, 2]}, index=["2018 Sales", "2019 Sales"]
      )
      ```
  - Series
    - It is a sequence of data values.
      ```python
      pd.Series([30, 50, 21])
      ```
    - In essence, it is a single column of a DataFrame.
    - You can assign column values to the Series the same way as before, using an `index` parameter. However, a Series does not have a column name, it only has one overall `name`.
      ```python
      pd.Series(
          [30, 50, 21], index=["2017 Sales", "2018 Sales", "2019 Sales"], name="Apples"
      )
      ```
- Reading Data Files
  - Data can be stored in any of a number of different forms and formats. By far the most basic of these is the humble CSV file. A CSV file is a table of values separated by commas.
    ```python
    # load data
    wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")

    # data dimention
    wine_reviews.shape
    >>> (129971, 14)

    # top rows
    wine_reviews.head()

    # bottom rows
    wine_reviews.tail()
    ```
  - The `pd.read_csv()` function has over 30 optional parameters.
  - For example, if your CSV file has a built-in index, pandas can use that column for the index (instead of creating a new one automatically).
    ```python
    wine_reviews = pd.read_csv(
        "../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0
    )
    ```
- Writing Data to File
  ```python
  animals = pd.DataFrame(
      {"Cows": [12, 20], "Goats": [22, 19]}, index=["Year 1", "Year 2"]
  )
  animals.to_csv("cows_and_goats.csv")
  ```

### [Indexing, Selecting & Assigning](https://www.kaggle.com/residentmario/indexing-selecting-assigning)
Pro data scientists do this dozens of times a day. You can, too!

- Naive Accessors
  - In Python, we can access the property of an object by accessing it as an attribute. A `reviews` object, might have a `country` property, which we can access by calling `reviews.country`. Columns in a pandas DataFrame work in much the same way.
  - If we have a Python dictionary, we can access its values using the indexing `[]` operator.
    ```python
    # select the `country` column
    reviews["country"]
    ```
  - A pandas Series looks kind of like a dictionary. So, to drill down to a single specific value, we need only use the indexing operator `[]` once more.
    ```python
    # select the first value from the `country` column
    reviews["country"][0]
    >>> 'Italy'
    ```
- Indexing in Pandas
  - For more advanced operations, pandas has its own accessor operators, `iloc` and `loc`.
  - Index-based Selection
    - Selecting data based on its **numerical position** in the data, like a matrix.
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
  - Label-based Selection
    - selecting data based on its **index value**, not its position.
    - **Inclusive** range.
      ```python
      # select the first value from the `country` column
      reviews.loc[0, "country"]

      # select all the entries from three specific columns
      reviews.loc[:, ["taster_name", "taster_twitter_handle", "points"]]
      ```
  - **Inclusive range**: `iloc` vs. `loc`
    ```python
    # select first three rows
    reviews.iloc[:3]
    # or
    reviews.loc[:2]
    
    # select the first 100 records of the `country` and `variety` columns.
    cols_idx = [0, 11]
    reviews.iloc[:100, cols_idx]
    # or
    cols = ["country", "variety"]
    reviews.loc[:99, cols]
    ```
  - Manipulating the Index
    ```python
    reviews.set_index("title")
    ```
- **Conditional Selection**
  - To do interesting things with the data, we often need to ask questions based on conditions.
  - To combine multiple conditions in pandas, **bitwise operators** must be used.
    ```python
    &    # AND          x & y
    |    # OR           x | y
    ^    # XOR          x ^ y
    ~    # NOT          ~x
    >>   # right shift  x>>
    <<   # left shift   x<<
    ```
  - For example, suppose that we're interested in better-than-average wines produced in Italy.
    ```python
    cond1 = (reviews["country"] == "Italy")
    cond2 = (reviews["points"] >= 90)
    reviews.loc[cond1 & cond2]
    ```
  - Built-in Conditional Selectors
    - `isin()` lets you select data whose value "is in" a list of values.
      ```python
      # select wines only from Italy or France
      reviews.loc[reviews.country.isin(["Italy", "France"])]
      ```
    - `isnull()` and `notnull()` let you highlight values which are (or are not) empty (NaN).
      ```python
      # filter out wines lacking a price tag in the dataset
      reviews.loc[reviews["price"].notnull()]
      ```
- Assigning Data
  ```python
  # you can assign either a constant value
  reviews["critic"] = "everyone"
  
  # or with an iterable of values
  reviews["index_backwards"] = range(len(reviews), 0, -1)
  ```

### [Summary Functions and Maps](https://www.kaggle.com/residentmario/summary-functions-and-maps)
Extract insights from your data.

- Summary Functions
  ```python
  # get summary statistic about a DataFrame
  reviews.describe()
  
  # get summary statistic about a Series
  reviews["points"].describe()
  
  # see the mean
  reviews["points"].mean()

  # see a list of unique values
  reviews["points"].unique()

  # see a list of unique values and how often they occur
  reviews["points"].value_counts()

  # get the title of the wine with the highest points-to-price ratio
  max_p2pr = (reviews["points"] / reviews["price"]).idxmax()
  reviews.loc[max_p2pr, "title"]
  >>> 'Bandit NV Merlot (California)'
  ```
- Maps
  - A function that takes one set of values and "maps" them to another set of values, for creating new representations from existing data.
  - `map()`
    ```python
    # remean the scores the wines received to 0
    review_points_mean = reviews["points"].mean()
    reviews["points"].map(lambda p: p - review_points_mean)

    # create a series counting how many times each of "tropical" or "fruity" appears in the description column
    n_tropical = reviews["description"].map(lambda desc: "tropical" in desc).sum()
    n_fruity = reviews["description"].map(lambda desc: "fruity" in desc).sum()
    pd.Series([n_tropical, n_fruity], index=["tropical", "fruity"])
    ```
    - The function you pass to `map()` should expect a single value from the Series (a point value, in the above example), and return a transformed version of that value. `map()` returns a new **Series**.
  - `apply()`
    ```python
    # remean the scores the wines received to 0
    def remean_points(row):
        row["points"] = row["points"] - review_points_mean
        return row
    reviews.apply(remean_points, axis="columns")

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
    - `apply()` is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row. `apply()` returns a new **DataFrame**.
    - If we had called `reviews.apply()` with `axis="index"`, then instead of passing a function to transform each row, we would need to give a function to transform each column.
  - Pandas built-ins Common Mapping Operators
    - They perform a simple operation between a lot of values on the left and a single (a lot of) value(s) on the right.
      ```python
      # remean the scores the wines received to 0
      review_points_mean = reviews["points"].mean()
      reviews["points"] - review_points_mean

      # combine country and region information in the dataset
      reviews["country"] + " - " + reviews["region_1"]
      ```
    - These operators are **faster** than `map()` or `apply()` because they uses speed ups built into pandas. All of the standard Python operators (`>`, `<`, `==`, and so on) work in this manner.
    - However, they are **not as flexible as** `map()` or `apply()`, which can do more advanced things, like applying conditional logic, which cannot be done with addition and subtraction alone.

### [Grouping and Sorting](https://www.kaggle.com/residentmario/grouping-and-sorting)
Scale up your level of insight. The more complex the dataset, the more this matters

- Groupwise Analysis
  - Maps allow us to transform data in a DataFrame or Series one value at a time for an entire column.
  - However, often we want to group our data, and then do something specific to the group the data is in.
  - `groupby()`
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
  - Aggregate Different Functions Simultaneously
    ```python
    # get a dataframe whose index is the variety category and values are the `min` and `max` prices
    reviews.groupby("variety")["price"].agg([min, max])
    ```
  - Multi-indexes, Group by More than One Column
    ```python
    # pick out the best wine by country and province
    reviews.groupby(["country", "province"]).apply(lambda df: df.loc[df["points"].idxmax()])
    ```
    - Multi-indices have several methods for dealing with their tiered structure which are absent for single-level indices.
    - They also require two levels of labels to retrieve a value.
    - The use cases for a multi-index are detailed alongside instructions on using them in the [MultiIndex / Advanced Selection](https://pandas.pydata.org/pandas-docs/stable/advanced.html) section of the pandas documentation.
    ```python
    # convert back to a regular index
    count_prov_best.reset_index()
    ```
- Sorting
  - `sort_values()`
    ```python
    # sort (country, province) based on how many reviews are belong to
    count_prov_reviewed = reviews.groupby(["country", "province"])["description"].agg([len])
    count_prov_reviewed.reset_index().sort_values(by="len", ascending=False)
    ```
  - Sort by More than One Column
    ```python
    count_prov_reviewed.reset_index().sort_values(by=["country", "len"], ascending=False)
    ```
  - `sort_index()`
    ```python
    # get a series whose index is wine prices and values is the maximum points a wine costing that much was given in a review. sort the values by price, ascending
    reviews.groupby("price")["points"].max().sort_index(ascending=True)
    ```

### [Data Types and Missing Values](https://www.kaggle.com/residentmario/data-types-and-missing-values)
Deal with the most common progress-blocking problems

- Dtypes
  - The data type for a column in a DataFrame or a Series is known as the `dtype`.
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
- Missing Values
  - `isnull()`, `notnull()`
    ```python
    # get a series of True & False, based on where NaNs are
    reviews["price"].isnull()
    
    # find the number of NaNs
    reviews["price"].isnull().sum()

    # create a dataframe of rows with missing (NaN) country
    reviews[reviews["country"].isnull()]
    ```
  - `fillna()`
    ```python
    # fill NaNs with Unknown
    reviews["region_1"].fillna("Unknown")
    ```
  - `replace()`
    ```python
    # replace missing data which is given some kind of sentinel values
    reviews["region_1"].replace(["Unknown", "Undisclosed", "Invalid"], "NaN")
    ```

### [Renaming and Combining](https://www.kaggle.com/residentmario/renaming-and-combining)
Data comes in from many sources. Help it all make sense together

- Renaming
  - `rename()`
    ```python
    # change the names of columns
    reviews.rename(columns={"region_1": "region", "region_2": "locale"})

    # change the indices of rows
    reviews.rename(index={0: "firstEntry", 1: "secondEntry"})

    # change the names of axes, form rows to wines, from columns to fields
    reviews.rename_axis("wines", axis="rows").rename_axis("fields", axis="columns")
    ```
- Combining
  - We will sometimes need to combine different DataFrames and/or Series. Pandas has three core methods for doing this. In order of increasing complexity, these are:
  - `concat()`
    - It will smush a list of elements together along an axis.
    - This is useful when we have data in different DataFrame or Series objects but having the same columns.
      ```python
      canadian_yt = pd.read_csv("../input/youtube-new/CAvideos.csv")
      british_yt = pd.read_csv("../input/youtube-new/GBvideos.csv")
      pd.concat([canadian_yt, british_yt])
      ```
  - `join()`
    - It lets you combine different DataFrame objects which have an index in common.
      ```python
      # pull down videos that happened to be trending on the same day in both Canada and the UK
      left = canadian_yt.set_index(["title", "trending_date"])
      right = british_yt.set_index(["title", "trending_date"])
      left.join(right, lsuffix="_CAN", rsuffix="_UK")
      ```
    - The `lsuffix` and `rsuffix` parameters are necessary when the data has the same column names in both datasets.
  - `merge()`

## **Data Visualization**
Make great data visualizations. A great way to see the power of coding!

### [Line Charts](https://www.kaggle.com/alexisbcook/line-charts)
Visualize trends over time

- Set up the notebook
  ```python
  import pandas as pd
  pd.plotting.register_matplotlib_converters()
  import matplotlib.pyplot as plt
  %matplotlib inline
  import seaborn as sns
  ```
- Line Chart
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

### [Bar Charts and Heatmaps](https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps)
Use color or length to compare categories in a dataset

- Bar Chart
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
  - **Important Note**: You must select the indexing column with `flight_data.index`, and it is not possible to use `flight_data['Month']`, because when we loaded the dataset, the `"Month"` column was used to index the rows.
- Heatmap
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

### [Scatter Plots](https://www.kaggle.com/alexisbcook/scatter-plots)
Leverage the coordinate plane to explore relationships between variables

### [Distributions](https://www.kaggle.com/alexisbcook/distributions)
Create histograms and density plots

### [Choosing Plot Types and Custom Styles](https://www.kaggle.com/alexisbcook/choosing-plot-types-and-custom-styles)
Customize your charts and make them look snazzy

### [Final Project](https://www.kaggle.com/alexisbcook/final-project)
Practice for real-world application

### [Creating Your Own Notebooks](https://www.kaggle.com/alexisbcook/creating-your-own-notebooks)
How to put your new skills to use for your next personal or work project

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
