# Data Scientist's Roadmap

A minimalist roadmap to The Data Science World, based on [Kaggle's Roadmap](https://www.kaggle.com/learn/overview)

## **Python**

Learn the most important language for data science.

### [Hello, Python](https://www.kaggle.com/colinmorris/hello-python)

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
          print("{} begins with \"{}\"".format(planet, initial))
      ```

### [Working with External Libraries](https://www.kaggle.com/colinmorris/working-with-external-libraries)

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

- Introduction
  > You ask your cousin how he's predicted real estate values in the past. and he says it is just intuition. But more questioning reveals that he's identified price patterns from houses he has seen in the past, and he uses those patterns to make predictions for new houses he is considering. Machine learning works the same way.
  - **Fitting** or **Training**: Capturing patterns from **training data**
  - **Predicting**: Getting results from applying the model to **new data**
- Decision Tree

### [Basic Data Exploration](https://www.kaggle.com/dansbecker/basic-data-exploration)

- Using Pandas to Get Familiar with the Data
  - **DataFrame**: The most important part of the Pandas library, similar to a sheet in Excel, or a table in a SQL database
    ```python
    # save filepath to variable for easier access
    melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
    # read the data and store data in DataFrame titled melbourne_data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # print a summary of the data in Melbourne data
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
    y = dropna_melbourne_data['Price']
    ```
  - Choosing **Features** (input columns, `X`)
    ```python
    feature_list = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
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
    from sklearn.tree import DecisionTreeRegressor
    # define model, `random_state` ensures you get the same results in each run
    melbourne_model = DecisionTreeRegressor(random_state=1)
    # fit model
    melbourne_model.fit(X, y)
    # make predictions
    predictions = melbourne_model.predict(X)
    ```

### [Model Validation](https://www.kaggle.com/dansbecker/model-validation)

- Summarizing the Model Quality into Metrics
  - There are many metrics for summarizing the model quality.
  - **Predictive Accuracy**: Will the model's predictions be close to what actually happens?
    - **Mean Absolute Error** (MAE)
      ```python
      from sklearn.metrics import mean_absolute_error
      mean_absolute_error(y, predictions)
      >>> 434.715
      ```
- **Big Mistake**: Measuring scores with the training data or the problem with **"In-Sample" scores**!
- **Validation Data**
  - **Making Predictions on New Data**
  - The most straightforward way to do that is to exclude some data from the model-building process, and then use those to test the model's accuracy.
    ```python
    from sklearn.model_selection import train_test_split
    # split data into training and validation data, for both features and target
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    # define model
    melbourne_model = DecisionTreeRegressor(random_state=1)
    # fit model
    melbourne_model.fit(train_X, train_y)
    # get predicted prices on validation data
    val_predictions = melbourne_model.predict(val_X)
    mean_absolute_error(val_y, val_predictions)
    >>> 259556.721
    ```
    > The MAE for the in-sample data was about 500 dollars. For out-of-sample data, it's more than 250,000 dollars. As a point of reference, the average home value in the validation data is 1.1 million dollars. So the error in new data is about a quarter of the average home value.
- There are many ways to improve a model, such as
  - Finding **better features**, the iterating process of building models with different features and comparing them to each other
  - Finding **better model types**
  - Finding **better data pre-processing methods**. For example look at the different ways of using `dropna()`
    ```python
    # raw data
    melbourne_data.shape
    >>> (13580, 21)
    # rows with price
    melbourne_data['Price'].dropna(axis=0).shape
    >>> (13580,)
    # rows with features we want
    melbourne_data[feature_list].dropna(axis=0).shape
    >>> (6858, 7)
    # rows without missing data
    melbourne_data.dropna(axis=0).shape
    >>> (6196, 21)
    ```

### [Underfitting and Overfitting](https://www.kaggle.com/dansbecker/underfitting-and-overfitting)

- Experimenting with Different Models
  - **Over-fitting**: Capturing spurious patterns that won't recur in the future, leading to less accurate predictions.
  - **Under-fitting**: Failing to capture relevant patterns, again leading to less accurate predictions.
  - In the **Decision Tree** model, the most important option to control the accuracy is the **tree's depth**, a measure of how many splits it makes before coming to a prediction.
    - A **deep tree** makes leaves with fewer objects. It causes **over-fitting**.
    - A **shallow tree** makes big groups. It causes **under-fitting**.
    - There are a few options for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes. But the `max_leaf_nodes` argument provides a very sensible way to control overfitting vs underfitting.
      ```python
      def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
          model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
          model.fit(train_X, train_y)
          val_predictions = model.predict(val_X)
          mae = mean_absolute_error(val_y, val_predictions)
          return(mae)
      # compare MAE with differing values of max_leaf_nodes
      candidate_max_leaf_nodes = [5, 50, 500, 5000]
      for max_leaf_nodes in candidate_max_leaf_nodes:
          mae_now = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
          print(f"Max Leaf Nodes: {max_leaf_nodes}  \t\t Mean Absolute Error: {mae_now}")
      ```
    - The lowest number is the optimal number of leaves.
      ```python
      # find the optimal number with a dict comprehension
      scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
      best_tree_size = min(scores, key=scores.get)
      ```

### [Random Forests](https://www.kaggle.com/dansbecker/random-forests)

- Introduction
  - Decision trees leave you with a difficult decision. A deep tree and over-fitting vs. a shallow one and under-fitting. Even today's most sophisticated modeling techniques face this tension. But, many models have clever ideas that can lead to better performance.
- A **Random Forest** model uses many trees, and makes a prediction by averaging the predictions of each component. It generally has much better predictive accuracy even with than a single decision tree, even with default parameters, without tuning the parameters like `max_leaf_nodes`.
  ```python
  # specify & fit model and make predictions
  from sklearn.ensemble import RandomForestRegressor
  forest_model = RandomForestRegressor(random_state=1)
  forest_model.fit(train_X, train_y)
  melb_preds = forest_model.predict(val_X)
  # calculate MAE
  from sklearn.metrics import mean_absolute_error
  mean_absolute_error(val_y, melb_preds)
  >>> 202888.181
  ```
  - The result is much better than that was before (259556.721).
- Some models, like the **XGBoost** model, provides better performance when tuned well with the right parameters (but which requires some skill to get the right model parameters).

### [Exercise: Machine Learning Competitions](https://www.kaggle.com/kernels/fork/1259198)

## **Intermediate Machine Learning**

Learn to handle missing values, non-numeric values, data leakage and more. Your models will be more accurate and useful.

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
