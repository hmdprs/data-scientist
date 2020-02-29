# Kaggle Learn

These micro-courses are the single fastest way to gain the skills you'll need to do independent data science projects.

## Python

Learn the most important language for data science.

* [x] [Hello, Python](https://www.kaggle.com/colinmorris/hello-python)
  - Hello, Python!
    - Variable Assignment
    - Function Calls
      - `print("viking_song " * 4)`
  - Numbers and Arithmetic in Python
    |          |                |
    | -------- | -------------- |
    | `a / b`  | true division  |
    | `a // b` | floor division |
    | `a % b`  | modulus        |
    | `a ** b` | exponentiation |
  - Order of Operators: **PEMDAS**
  - Builtin Functions for Working with Numbers
    - `min()`
    - `max()`
    - `abs()`
    - `int()`
    - `float()`
* [x] [Functions and Getting Help](https://www.kaggle.com/colinmorris/functions-and-getting-help)
  - Getting Help
    - `help()`
  - Defining Functions
  - Docstrings
    - info of a func between `"""  """`, returns by `help()`
  - Functions that don't Return
    - `print()`
  - Default Arguments
    - `print(..., sep='\t')`
  - Functions Applied to Functions
    - `fn(fn(arg))`
* [x] [Booleans and Conditionals](https://www.kaggle.com/colinmorris/booleans-and-conditionals)
  - Booleans
    - Comparison Operations
      |          |                              |
      | -------- | ---------------------------- |
      | `a == b` | a equal to b                 |
      | `a != b` | a not equal to b             |
      | `a < b`  | a less than b                |
      | `a > b`  | a greater than b             |
      | `a <= b` | a less than or equal to b    |
      | `a >= b` | a greater than or equal to b |
    - Combining Boolean Values
      - Order of Operators:
        - `()`
        - `**`
        - `+x, -x, ~x`
        - `*, /, //, %`
        - `+, -`
        - `<<, >>`
        - `&`
        - `^`
        - `|`
        - `==, !=, >, >=, <, <=, is, is not, in, not in`
        - `not`
        - `and`
        - `or`
      - use `()` for clarity & readability
  - Conditionals
    - `if`
    - `elif`
    - `else`
  - Boolean Conversion
    - `bool()`
    - all numbers are treated as true, except `0`
    - all strings are treated as true, except the empty string `""`
    - empty sequences (strings, lists, tuples, sets)  are "falsey" and the rest are "truthy"
  - Conditional Expressions (aka 'ternary')
    - Setting a variable to either of two values depending on some condition
    - 1-line-form:
      - `outcome = 'failed' if grade < 50 else 'passed'`
* [x] [Lists](https://www.kaggle.com/colinmorris/lists)
  - Lists
    - `[]` + mutable + a mix of different types of variables
  	- Indexing
    	- `0` is first, `-1` is last
    - Slicing
      - `planets[:3]`
      - `planets[-3:]`
    - Changing Lists
      - `planets[:3] = ['Mur', 'Vee', 'Ur']`
    - List Functions
      - `len()`
      - `sorted()`
      - `max()`
      - `sum()`
      - `any()`
    - Methods
      - `c = 12+5j`
      - `c.imag`
      - `c.real`
      - `x = 12`
      - `x.bit_length()`
      - `help(x.bit_length)`
      - `help(int.bit_length)`
    - List Methods
      - `list.append()`
      - `help(list.append)`
      - `list.pop()`
      - `list.index()`
      - `in`
      - to find all methods: `help(list)`
  - Tuples
    - `()` or `, , ,` + immutable
    - `x = 0.125`
    - `numerator, denominator = x.as_integer_ratio()`
* [x] [Loops and List Comprehensions](https://www.kaggle.com/colinmorris/loops-and-list-comprehensions)
  - Loops
    - `for _ in _:`
  	- in every iteratable objects: list, tuples, strings, ...
  	- `range()`
  	- `while` loops
  - List Comprehensions
    - `squares = [n**2 for n in range(10)]`
  	- with `if`:
    	- `short_planets = [planet.upper() + "!" for planet in planets if len(planet) < 6]`
  	- `[32 for planet in planets]`
  	- with functions like `min()`, `max()`, `sum()`, `any()`:
    	- `return len([num for num in nums if num < 0])`
    	- `return sum([num < 0 for num in nums])`
    	- `return any([num % 7 == 0 for num in nums])`
  - Solving a problem with less code is always nice, but it's worth keeping in mind the following lines from **The Zen of Python** (`import this`):
    > Readability counts.<br>
    > Explicit is better than implicit.
* [ ] [String and Directories](https://www.kaggle.com/colinmorris/strings-and-dictionaries)
* [ ] [Working with External Libraries](https://www.kaggle.com/colinmorris/working-with-external-libraries)

## Intro to Machine Learning

Learn the core ideas in machine learning, and build your first models.

## Intermediate Machine Learning

Learn to handle missing values, non-numeric values, data leakage and more. Your models will be more accurate and useful.

## Data Visualization

Make great data visualizations. A great way to see the power of coding!

## Pandas

Solve short hands-on challenges to perfect your data manipulation skills.

## Feature Engineering

Discover the most effective way to improve your models.

## Deep Learning

Use TensorFlow to take machine learning to the next level. Your new skills will amaze you.

## Intro to SQL

Learn SQL for working with databases, using Google BigQuery to scale to massive datasets.

## Advanced SQL

Take your SQL skills to the next level.

## Geopatial Analysis

Create interactive maps, and discover patterns in geospatial data.

## Microchallenges

Solve ultra-short challenges to build and test your skill.

## Machine Learning Explainability

Extract human-understandable insights from any machine learning model.

## Natural Language Processing

Distinguish yourself by learning to work with text data.