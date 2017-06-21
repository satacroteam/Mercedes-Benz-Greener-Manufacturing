# -*- coding: utf-8 -*-
import multiprocessing as mp
# mp.set_start_method('forkserver')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import chi2_kernel
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

train = pd.read_csv('train.csv')
train = train.set_index(train.ID).drop('ID', axis=1)

test = pd.read_csv('test.csv')
test = test.set_index(test.ID).drop('ID', axis=1)

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


cat_features= ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

# cols_binary = train.drop(['y']+cat_features, axis=1).keys()
# for i in range(len(cols_binary)):
# 	col_to_drop = cols[i]
# 	print(col_to_drop)
# 	duplicates = train.drop(['y', col_to_drop], axis=1).duplicated(keep=False)
# 	df = train[duplicates].drop('y', axis=1)
# 	df2 = df.sort_values(by=df.columns.tolist())
# 	dfs_index = {i-1: train.loc[g.index] for i,g in df2.groupby((~((df2 == df2.shift(1)).all(1))).cumsum())}
# 	for frame in dfs_index:
# 		gb = dfs_index[frame].groupby(col_to_drop)
# 		dict_gb = dict(list(gb))
# 		if len(dict_gb)>1:
# 			mean0 = np.mean(dict_gb[0].y)
# 			mean1 = np.mean(dict_gb[1].y)
# 			print(mean0, mean1)
#
#
# # 2 variables
# for i in range(len(cols_binary)):
# 	col_to_drop = cols_binary[i]
# 	for j in range(i + 1, len(cols_binary)):
# 		col_to_drop_2 = cols_binary[j]
# 		print(col_to_drop, col_to_drop_2)
# 		duplicates = train.drop(['y', col_to_drop, col_to_drop_2], axis=1).duplicated(keep=False)
# 		df = train[duplicates].drop(['y', col_to_drop, col_to_drop_2], axis=1)
# 		df2 = df.sort_values(by=df.columns.tolist())
# 		dfs_index = {i - 1: train.loc[g.index] for i, g in df2.groupby((~((df2 == df2.shift(1)).all(1))).cumsum())}
# 		for frame in dfs_index:
# 			df_temp = train.loc[dfs_index[frame].index][[col_to_drop, col_to_drop_2, 'y']]
# 			if any(~df_temp[[col_to_drop, col_to_drop_2]].duplicated(keep=False)):
# 				print(df_temp)
#
# Columns Duplicates
# cols_binary = train.drop(['y']+cat_features, axis=1).keys()
# train_col_dupl = train.drop(['y']+ cat_features, axis=1)
#
# keys_test = train_col_dupl.keys( )
# for key in keys_test:
# 	if key in train_col_dupl.keys():
# 		print(key)
# 		col_test = 	train_col_dupl[key]
# 		for col in train_col_dupl:
# 			if col != key:
# 				if col_test.equals(train_col_dupl[col]) | (col_test!=1).equals((train_col_dupl[col]==1)):
# 					train_col_dupl.drop(col, axis=1, inplace=True)


plt.scatter(train.index, train.y, s=10)
remaining_cols = list(np.array(['X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18',
       'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X26', 'X27', 'X28',
       'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X36', 'X38', 'X40',
       'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49',
       'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58',
       'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67',
       'X68', 'X69', 'X70', 'X71', 'X73', 'X74', 'X75', 'X77', 'X78',
       'X79', 'X80', 'X81', 'X82', 'X83', 'X85', 'X86', 'X87', 'X88',
       'X89', 'X90', 'X91', 'X92', 'X95', 'X96', 'X97', 'X98', 'X99',
       'X100', 'X101', 'X103', 'X104', 'X105', 'X106', 'X108', 'X109',
       'X110', 'X111', 'X112', 'X114', 'X115', 'X116', 'X117', 'X118',
       'X123', 'X124', 'X125', 'X126', 'X127', 'X128', 'X129', 'X131',
       'X132', 'X133', 'X135', 'X137', 'X138', 'X139', 'X140', 'X141',
       'X142', 'X143', 'X144', 'X145', 'X148', 'X150', 'X151', 'X152',
       'X153', 'X154', 'X155', 'X156', 'X159', 'X160', 'X161', 'X162',
       'X163', 'X164', 'X165', 'X166', 'X167', 'X168', 'X169', 'X170',
       'X171', 'X173', 'X174', 'X175', 'X176', 'X177', 'X178', 'X179',
       'X180', 'X181', 'X182', 'X183', 'X184', 'X185', 'X186', 'X187',
       'X189', 'X190', 'X191', 'X192', 'X195', 'X196', 'X197', 'X198',
       'X200', 'X201', 'X202', 'X203', 'X204', 'X206', 'X207', 'X208',
       'X209', 'X210', 'X211', 'X212', 'X215', 'X217', 'X218', 'X219',
       'X220', 'X221', 'X223', 'X224', 'X225', 'X228', 'X229', 'X230',
       'X231', 'X234', 'X236', 'X237', 'X238', 'X240', 'X241', 'X246',
       'X249', 'X250', 'X251', 'X252', 'X255', 'X256', 'X257', 'X258',
       'X259', 'X260', 'X261', 'X264', 'X265', 'X267', 'X269', 'X270',
       'X271', 'X272', 'X273', 'X274', 'X275', 'X276', 'X277', 'X278',
       'X280', 'X281', 'X282', 'X283', 'X284', 'X285', 'X286', 'X287',
       'X288', 'X291', 'X292', 'X294', 'X295', 'X298', 'X300', 'X301',
       'X304', 'X305', 'X306', 'X307', 'X308', 'X309', 'X310', 'X311',
       'X312', 'X313', 'X314', 'X315', 'X316', 'X317', 'X318', 'X319',
       'X321', 'X322', 'X323', 'X325', 'X327', 'X328', 'X329', 'X331',
       'X332', 'X333', 'X334', 'X335', 'X336', 'X337', 'X338', 'X339',
       'X340', 'X341', 'X342', 'X343', 'X344', 'X345', 'X346', 'X348',
       'X349', 'X350', 'X351', 'X352', 'X353', 'X354', 'X355', 'X356',
       'X357', 'X358', 'X359', 'X361', 'X362', 'X363', 'X366', 'X367',
       'X368', 'X369', 'X370', 'X371', 'X372', 'X373', 'X374', 'X375',
       'X376', 'X377', 'X378', 'X379', 'X380', 'X383', 'X384'], dtype=object))
good_cols = cat_features + remaining_cols

train = train[['y']+good_cols]

merged = pd.concat((train[good_cols], test[good_cols]))

# Hot encoder
def one_hot(frame, cols):
	df = frame.copy()
	for each in cols:
		dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
		df.drop(each, axis=1, inplace=True)
		df = pd.concat([df, dummies], axis=1)
	return df

merged_encoded = one_hot(merged, cat_features)
train_to_encode = merged_encoded.loc[train.index]


## Autoencoder
from keras.layers import Input, Dense, regularizers
from keras.models import Model

feature_to_encode = np.array(merged_encoded)

# Params
encoding_dim = 32
imput_dim = feature_to_encode.shape[1]

# input
input = Input(shape=(imput_dim,))

# encoder
encoded = Dense(128, activation='relu')(input)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(imput_dim, activation='sigmoid')(decoded)

# input to input reconstructed
autoencoder = Model(input, decoded)

# Encoder
encoder = Model(input, encoded)

# Compile and fit
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(feature_to_encode, feature_to_encode,
                epochs=50,
                batch_size=50,
                shuffle=True)


# encode and decode features
train_encoded_auto = np.array(train_to_encode)
encoded_features = encoder.predict(train_encoded_auto)
keep_indices = []
for i in range(encoded_features.shape[1]):
	if sum(encoded_features[:, i]**2) > 2:
		keep_indices.append(i)
encoded_features = encoded_features[:, keep_indices]

# Plot
plot_size = 5
k = 0
l = 0
fig = plt.figure('0')
col_left = range(encoded_features.shape[1])
for i in col_left:
	for j in range(i, len(col_left)):
		if k > plot_size**2-1:
			k = 0
			l += 1
			fig = plt.figure(str(l))
		k += 1
		ax = fig.add_subplot(plot_size, plot_size, k)  # this line adds sub-axes
		plt.scatter(encoded_features[:len(train), i], encoded_features[:len(train), j], c=train.y, cmap=plt.cm.terrain, s=1)
		plt.suptitle('%s|%s'%(i,j))


# Classifier 87
import xgboost as xgb
from sklearn.metrics import classification_report
y_87 = train.y < 87
clf = xgb.XGBClassifier()
clf.fit(encoded_features, y_87)
preds_87 = clf.predict(encoded_features)
print(classification_report(y_87, preds_87))
