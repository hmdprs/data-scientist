# data fields

# -----target
# SalePrice - the property's sale price in dollars.

# -----variables
# numeric
features_numeric = [
    "MSSubClass",  # The building class [20-190]
    "LotArea",  # Lot size in square feet [1300-215000]
    "BsmtFinSF1",  # Type 1 finished square feet [0-5644]
    "BsmtFinSF2",  # Type 2 finished square feet [0-1474]
    "BsmtUnfSF",  # Unfinished square feet of basement area [0-2336]
    "TotalBsmtSF",  # Total square feet of basement area [0-6110]
    "1stFlrSF",  # First Floor square feet [334-4690]
    "2ndFlrSF",  # Second floor square feet [0-2690]
    "LowQualFinSF",  # Low quality finished square feet (all floors) [0-572]
    "GrLivArea",  # Above grade (ground) living area square feet [334-5642]
    "GarageArea",  # Size of garage in square feet [0-1418]
    "WoodDeckSF",  # Wood deck area in square feet [0-857]
    "OpenPorchSF",  # Open porch area in square feet [0-547]
    "EnclosedPorch",  # Enclosed porch area in square feet [0-552]
    "3SsnPorch",  # Three season porch area in square feet [0-508]
    "ScreenPorch",  # Screen porch area in square feet [0-480]
    "PoolArea",  # Pool area in square feet [0-738]
    "MiscVal",  # $Value of miscellaneous feature [0-15500]
]  # 9,481

# categorical
features_categorical = [
    "MSZoning",  # The general zoning classification (79%, 15%)
    "LotShape",  # General shape of property (63%, 33%)
    "LotConfig",  # Lot configuration (72%, 18%)
    "Neighborhood",  # Physical locations within Ames city limits (15%, 10%)
    "Condition1",  # Proximity to main road or railroad (86%, 6%)
    "BldgType",  # Type of dwelling (84%, 8%)
    "HouseStyle",  # Style of dwelling (50%, 30%)
    "OverallQual",  # Overall material and finish quality [1-10]
    "OverallCond",  # Overall condition rating [1-9]
    "RoofStyle",  # Type of roof (78%, 20%)
    "Exterior1st",  # Exterior covering on house (35%, 15%)
    "Exterior2nd",  # Exterior covering on house (if more than one material) (35%, 15%)
    "MasVnrType",  # Masonry veneer type (59%, 30%)
    "MasVnrArea",  # Masonry veneer area in square feet (59%, 1%)
    "ExterQual",  # Exterior material quality (62%, 33%)
    "ExterCond",  # Present condition of the material on the exterior (88%, 10%)
    "Foundation",  # Type of foundation (44%, 43%)
    "BsmtQual",  # Height of the basement (44%, 42%)
    "BsmtExposure",  # Walkout or garden level basement walls (65%, 15%)
    "BsmtFinType1",  # Quality of basement finished area (29%, 29%)
    "BsmtFinType2",  # Quality of second finished area (if present) (86%, 4%)
    "HeatingQC",  # Heating quality and condition (51%, 29%)
    "CentralAir",  # Central air conditioning
    "BsmtFullBath",  # Basement full bathrooms [0-3]
    "BsmtHalfBath",  # Basement half bathrooms [0-2]
    "FullBath",  # Full bathrooms above grade [0-3]
    "HalfBath",  # Half baths above grade [0-2]
    "BedroomAbvGr",  # Number of bedrooms above basement level [0-8]
    "KitchenAbvGr",  # Number of kitchens [0-3]
    "KitchenQual",  # Kitchen quality (50%, 40%)
    "TotRmsAbvGrd",  # Total rooms above grade (does not include bathrooms) [2-14]
    "Fireplaces",  # Number of fireplaces [0-3]
    "FireplaceQu",  # Fireplace quality (47%, 26%)
    "GarageType",  # Garage location (60%, 27%)
    "GarageFinish",  # Interior finish of the garage (41%, 29%)
    "GarageCars",  # Size of garage in car capacity [0-4]
    "SaleType",  # Type of sale (87%, 8%)
    "SaleCondition",  # Condition of sale (82%, 9%)
]  # You have categorical data, but your model needs something numerical.

# date
features_date = [
    "YearBuilt",  # Original construction date [1872-2010]
    "YearRemodAdd",  # Remodel date [1950-2010]
    "MoSold",  # Month Sold [1-12]
    "YrSold",  # Year Sold [2006-2010]
]  # mae = 22,972

# first guess non useful (>= 90%)
features_nonuseful = [
    "Street",  # Type of road access (100%, )
    "LandContour",  # Flatness of the property (90%, 4%)
    "Utilities",  # Type of utilities available (100%, )
    "LandSlope",  # Slope of property (95%, 4%)
    "Condition2",  # Proximity to main road or railroad (if a second is present) (99%, )
    "RoofMatl",  # Roof material (98%, )
    "BsmtCond",  # General condition of the basement (90%, 4%)
    "Heating",  # Type of heating (98%, 1%)
    "Electrical",  # Electrical system (91%, 6%)
    "Functional",  # Home functionality rating (93%, 3%)
    "GarageQual",  # Garage quality (90%, 6%)
    "GarageCond",  # Garage condition (91%, 6%)
    "PavedDrive",  # Paved driveway (92%, 6%)
]  # You have categorical data, but your model needs something numerical.

# NA
features_na = [
    "LotFrontage",  # Linear feet of street connected to property (18%NA, 10%)
    "Alley",  # Type of alley access (94%NA, )
    "GarageYrBlt",  # Year garage was built (6%NA, 4%)
    "PoolQC",  # Pool quality (100%NA, )
    "Fence",  # Fence quality (81%NA, 11%)
    "MiscFeature",  # Miscellaneous feature not covered in other categories (96%NA, 3%)
]

# used
features = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
]  # mae: 9,481

from sklearn.ensemble import RandomForestRegressor

used_model = RandomForestRegressor(random_state=1)

default_model = RandomForestRegressor(
    n_estimators=100,  # The number of trees in the forest.
    criterion="mse",  # The function to measure the quality of a split. ["mse", "mae"]
    max_depth=None,  # The maximum depth of the tree.
    min_samples_split=2,  # The minimum number of samples required to split an internal node.
    min_samples_leaf=1,  # The minimum number of samples required to be at a leaf node
    min_weight_fraction_leaf=0,  # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
    max_features="auto",  # The number of features to consider when looking for the best split.
    max_leaf_nodes=None,  # Grow trees with `max_leaf_nodes` in best-first fashion. Best nodes are defined as relative reduction in impurity.
    min_impurity_decrease=0,  # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    bootstrap=True,  # Whether bootstrap samples are used when building trees. If `False`, the whole datset is used to build each tree.
    oob_score=False,  # whether to use out-of-bag samples to estimate the R^2 on unseen data.
    n_jobs=None,  # The number of jobs to run in parallel. `None` means 1 and `-1` means using all processors.
    random_state=1,  # Controls both the randomness of the bootstrapping of the samples used when building trees.
    verbose=0,  # Controls the verbosity when fitting and predicting.
    warm_start=False,  # When set to `True`, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
    ccp_alpha=0.0,  # Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than `ccp_alpha` will be chosen.
    max_samples=None,  # If bootstrap is `True`, the number of samples to draw from X to train each base estimator.
)
