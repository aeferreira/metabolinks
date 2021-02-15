import pandas as pd
from metabolinks import add_labels
import metabolinks.datasets as datasets
import metabolinks.transformations as transformations


print('\nReading unlabeled sample data ------------\n')
dataset = datasets.demo_dataset('demo1').as_pandas()
#print(dataset)
print('-- info --------------')
print(dataset.cdf.info())
print('-- global info---------')
print(dataset.cdf.info(all_data=True))
print('-----------------------')

print('\nretrieving subset of data ----------')
print('--- sample s39 ----------')
asample = dataset.cdf.take(sample='s39')
print(asample)
assert type(asample) == pd.Series
assert asample[98.34894] == 2165052.0
print('--- samples s39 s33 ----------')
asample = dataset.cdf.take(sample=('s39', 's33'))
print(asample)
assert type(asample) == pd.DataFrame

print('\nReading dataset data with labels ------------\n')
dataset = datasets.demo_dataset('demo2').as_pandas()
#print(dataset)
print('-- info --------------')
print(dataset.cdl.info())
print('-- global info---------')
print(dataset.cdl.info(all_data=True))
print('-- label of s39 --------------')
print(dataset.cdl.label_of('s39'))
print('-- samples of l2 --------------')
print(dataset.cdl.samples_of('l2'))
print('-----------------------')


print('\nretrieving subsets of data ----------')
print('--- sample s39 ----------')
asample = dataset.cdl.take(sample='s39')
print(asample)
assert type(asample) == pd.Series

print('--- label l2 ----------')
asample = dataset.cdl.take(label='l2')
print(asample)
assert type(asample) == pd.DataFrame

print('\nretrieving features')
print('--- whole data ----------')
print(list(dataset.cdl.features()))
print('--- sample s39 ----------')
asample = dataset.cdl.features(sample='s39')
print(list(asample))
print('--- label l2 ----------')
asample = dataset.cdl.features(label='l2')
print(asample.values)

print('\nUsing subset_iloc to double label l2 ----')
newdataset = dataset.copy()
#print(newdataset)
print('\n-- label l2 is replaced by 2 x -------------')
double = newdataset.cdl.subset(label='l2') * 2
iloc = newdataset.cdl.subset_iloc(label='l2')
print(iloc)
newdataset.iloc[:, iloc] = double
print(newdataset)
print(newdataset.cdl.info())

print('\nUsing subset_loc to double label l2 ----')
newdataset = dataset.copy()
#print(newdataset)
# print('\n--original data with sorted column index -')
# newdataset = newdataset.sort_index(axis='columns')
# print(newdataset)
print('\n-- label l2 replaced by double -------------')
double = newdataset.cdl.subset(label='l2') * 2
loc = newdataset.cdl.subset_loc(label='l2')
print(loc)
newdataset.loc[:, loc] = double
print(newdataset)
print(newdataset.cdl.info())

print('\nUsing subset_where to double label l2 ----')
newdataset = dataset.copy()
#print(newdataset)
print('\n--original data with sorted column index -')
newdataset = newdataset.sort_index(axis='columns')
print(newdataset)
print('\n-- label l2 replaced by double -------------')
double = newdataset.cdl.subset(label='l2') * 2
bool_loc = newdataset.cdl.subset_where(label='l2')
newdataset = newdataset.mask(bool_loc, double)
#dataset[bool_loc] = double
print(newdataset)
print(newdataset.cdl.info())

print('\nData transformations using pipe ----')
print('--- using fillna_zero ----------')
trans = transformations.fillna_zero
new_data = dataset.cdl.pipe(trans)
print(new_data)
print('--- features using fillna_value ----------')
trans = transformations.fillna_value
new_data = dataset.cdl.pipe(trans, value=10).cdl.features().to_list()
print(new_data)

print('\nExisting labels ----')
print(dataset.cdl.unique_labels)

print('\nSetting new labels -- L1 L2 L3 --')
dataset.cdl.labels = ['L1', 'L2', 'L3']
#print(dataset)
print(dataset.cdl.info())

print('\nSetting new labels -- L1 --')
dataset.cdl.labels = 'L1'
#print(dataset)
print(dataset.cdl.info())

print('\nSetting new labels --- None -')
dataset.cdl.labels = None
#print(dataset)
print(dataset.cdl.info())

print('\nSetting new labels and samples ----')
print('--- labels L1 L2 L3 ----------')
dataset.cdl.labels = ['L1', 'L2', 'L3']
print('--- samples as default ----------')
dataset.cdl.samples = None
#print(dataset)
print(dataset.cdl.info())

print('\nReading again sample data with labels (as io stream) ------------\n')
dataset = datasets.demo_dataset('demo2').as_pandas()
print(dataset)
print('-- info --------------')
print(dataset.cdl.info())
print('-- global info---------')
print(dataset.cdl.info(all_data=True))
print('-----------------------')

print('\nTesting ms.erase_labels() ----')
dataset_unlabeled = dataset.cdl.erase_labels()
#print(dataset_unlabeled)
print('-- info --------------')
print(dataset_unlabeled.cdf.info())
print('-- global info---------')
print(dataset_unlabeled.cdf.info(all_data=True))
print('-----------------------')

print('\nretrieving subsets of data ----------')
print('--- sample s39 ----------')
asample = dataset_unlabeled.cdf.take(sample='s39')
print(asample)
assert type(asample) == pd.Series
#print(type(asample))
print('--- samples s38 s32 ----------')
asample = dataset_unlabeled.cdf.take(sample=['s38', 's32'])
print(asample)
assert type(asample) == pd.DataFrame
#print(type(asample))

print('\nretrieving features')
print('--- whole data ----------')
print(list(dataset_unlabeled.cdf.features()))
print('--- sample s39 ----------')
asample = dataset_unlabeled.cdf.features(sample='s39')
print(list(asample))

print('\nadding labels again')
print('--- adding L1, L2 ----------')
newdataset = add_labels(dataset, labels=['L1', 'L2'])
print(newdataset)


