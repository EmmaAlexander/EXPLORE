import explore_commons
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
print("starting")

# Set seed for reproducibility
np.random.seed(1606421)

# Read to dataframe
df, classes, header = explore_commons.read_data('/Users/emma/OneDrive/Work/MLstars/10percent/10percent_full.dat',
                                                '/Users/emma/OneDrive/Work/MLstars/10percent/10percent_labels.txt')

# Preprocess df
df, classes = explore_commons.process_data_frame(df, classes, header)

# Merge duplicate Wavelengths
df = explore_commons.merge_duplicate_wavelength_cols(df)

# Label encode
le, labels = explore_commons.label_encode_data(classes)

# Galactic Coordinate Conversion
df = explore_commons.convert_to_galactic_coords(df)

idx = pd.IndexSlice

# Traditionally Most Relevant Variables Only
BasePackage = df.loc[:, idx["Adopted", ["Teff", "Lum"], "Value", :, :, :, :]]

# Physically Irrelevant Variables Only
BiasPackage = df.loc[:, idx["Adopted", ["RA", "Dec", "PMRA", "PMDec", "Distance"], "Value", :, :, :, :]]

# Spectral Measurements Only
SpectraPackage = pd.concat([df.loc[:, idx["Photometry", :, "Error", :, :, :, :]],
                            df.loc[:, idx[["Model", "Dereddened"], :, "Value", :, :, :, :]]], axis=1)

# Transformed Spectra
SpectraPackage = explore_commons.transform_Spectra(df)

# All Physically Relevant Variables
PhysicsPackage = pd.concat([BasePackage, df.loc[:, idx["Adopted", ["E(B-V)", 'logg', '[Fe/H]'], "Value", :, :, :, :]],
                            df.loc[:, idx["Ancillary", "Tspec", "Value", :, :, :, :]], SpectraPackage], ignore_index=True, axis=1)

# All Relevant Variables
FullPackage = pd.concat([BiasPackage, PhysicsPackage], ignore_index=True, axis=1)

# Custom Train Test Split
train_X, train_y, test_X, test_y = explore_commons.custom_train_test_split(FullPackage, labels, test_size=0.2, min_class_instances=40)


xgb_1 = xgb.XGBClassifier()
final_1 = xgb_1.fit(train_X, train_y)
manual_accuracy_1 = metrics.accuracy_score(test_y, final_1.predict(test_X))
manual_fscore = metrics.f1_score(test_y, final_1.predict(test_X), average='macro')
print(manual_fscore)
print(manual_accuracy_1)
