
# Read file
train = pd.read_csv('train.csv')
train.drop('ID', axis=1, inplace=True)
test = pd.read_csv('test.csv')

# Remove constant columns
constcols_train = list(train.loc[:, train.apply(lambda i: len(i.unique()) == 1)].columns)
goodcols = np.sort(list(set(train.columns).intersection(set(test.columns)) - set(constcols_train)))

train = pd.concat((train.y, train[goodcols[np.argsort([int(a.split('X')[1]) for a in goodcols])]]), axis=1)
test = test[goodcols[np.argsort([int(a.split('X')[1]) for a in goodcols])]]
