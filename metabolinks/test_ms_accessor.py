from six import StringIO
import pandas as pd
from metabolinks import add_labels
import metabolinks.dataio as dataio
import metabolinks.datasets as datasets
import metabolinks.transformations as transformations


print('\nReading sample data (as io stream) ------------\n')
dataset = dataio.read_data_csv(StringIO(datasets.demo_data1()))
print(dataset)
print('-- info --------------')
print(dataset.ums.info())
print('-- global info---------')
print(dataset.ums.info(all_data=True))
print('-----------------------')

print('\nretrieving subset of data ----------')
print('--- sample s39 ----------')
asample = dataset.ums.take(sample='s39')
print(asample)
print(type(asample))
print(asample[98.34894])
print('--- samples s39 s33 ----------')
asample = dataset.ums.take(sample=('s39', 's33'))
print(asample)
print(type(asample))

print('\nReading sample data with labels (as io stream) ------------\n')
data = dataset = dataio.read_data_csv(StringIO(datasets.demo_data2()), has_labels=True)
print(dataset)
print('-- info --------------')
print(dataset.ms.info())
print('-- global info---------')
print(dataset.ms.info(all_data=True))
print('-- label of s39 --------------')
print(dataset.ms.label_of('s39'))
print('-- samples of l2 --------------')
print(dataset.ms.samples_of('l2'))
print('-----------------------')


print('\nretrieving subsets of data ----------')
print('--- sample s39 ----------')
asample = dataset.ms.take(sample='s39')
print(asample)
print(type(asample))
# print(asample[97.59185])
print('--- label l2 ----------')
asample = dataset.ms.take(label='l2')
print(asample)
print(type(asample))

print('\nretrieving features')
print('--- whole data ----------')
print(list(dataset.ms.features()))
print('--- sample s39 ----------')
asample = dataset.ms.features(sample='s39')
print(list(asample))
print('--- label l2 ----------')
asample = dataset.ms.features(label='l2')
print(asample.values)

print('\nUsing subset_iloc to double label l2 ----')
newdataset = dataset.copy()
print(newdataset)
print('\n-- label l2 replaced by double -------------')
double = newdataset.ms.subset(label='l2') * 2
iloc = newdataset.ms.subset_iloc(label='l2')
print(iloc)
newdataset.iloc[:, iloc] = double
print(newdataset)
print(newdataset.ms.info())

print('\nUsing subset_loc to double label l2 ----')
newdataset = dataset.copy()
print(newdataset)
# print('\n--original data with sorted column index -')
# newdataset = newdataset.sort_index(axis='columns')
# print(newdataset)
print('\n-- label l2 replaced by double -------------')
double = newdataset.ms.subset(label='l2') * 2
loc = newdataset.ms.subset_loc(label='l2')
print(loc)
newdataset.loc[:, loc] = double
print(newdataset)
print(newdataset.ms.info())

print('\nUsing subset_where to double label l2 ----')
newdataset = dataset.copy()
print(newdataset)
print('\n--original data with sorted column index -')
newdataset = newdataset.sort_index(axis='columns')
print(newdataset)
print('\n-- label l2 replaced by double -------------')
double = newdataset.ms.subset(label='l2') * 2
bool_loc = newdataset.ms.subset_where(label='l2')
newdataset = newdataset.mask(bool_loc, double)
#dataset[bool_loc] = double
print(newdataset)
print(newdataset.ms.info())

print('\nData transformations using pipe ----')
print('--- using fillna_zero ----------')
trans = transformations.fillna_zero
new_data = dataset.ms.pipe(trans)
print(new_data)
print('--- features using fillna_value ----------')
trans = transformations.fillna_value
new_data = dataset.ms.pipe(trans, value=10).ms.features().to_list()
print(new_data)

print('\nExisting labels ----')
print(dataset.ms.unique_labels)

print('\nSetting new labels -- L1 L2 L3 --')
dataset.ms.labels = ['L1', 'L2', 'L3']
print(dataset)
print(dataset.ms.info())

print('\nSetting new labels -- L1 --')
dataset.ms.labels = 'L1'
print(dataset)
print(dataset.ms.info())

print('\nSetting new labels --- None -')
dataset.ms.labels = None
print(dataset)
print(dataset.ms.info())

print('\nSetting new labels and samples ----')
print('--- labels L1 L2 L3 ----------')
dataset.ms.labels = ['L1', 'L2', 'L3']
print('--- samples as default ----------')
dataset.ms.samples = None
print(dataset)
print(dataset.ms.info())

print('\nReading again sample data with labels (as io stream) ------------\n')
dataset = dataio.read_data_csv(StringIO(datasets.demo_data2()), has_labels=True)
print(dataset)
print('-- info --------------')
print(dataset.ms.info())
print('-- global info---------')
print(dataset.ms.info(all_data=True))
print('-----------------------')

print('\nTesting ms.erase_labels() ----')
dataset_unlabeled = dataset.ms.erase_labels()
print(dataset_unlabeled)
print('-- info --------------')
print(dataset_unlabeled.ums.info())
print('-- global info---------')
print(dataset_unlabeled.ums.info(all_data=True))
print('-----------------------')

print('\nretrieving subsets of data ----------')
print('--- sample s39 ----------')
asample = dataset_unlabeled.ums.take(sample='s39')
print(asample)
print(type(asample))
print('--- samples s38 s32 ----------')
asample = dataset_unlabeled.ums.take(sample=['s38', 's32'])
print(asample)
print(type(asample))

print('\nretrieving features')
print('--- whole data ----------')
print(list(dataset_unlabeled.ums.features()))
print('--- sample s39 ----------')
asample = dataset_unlabeled.ums.features(sample='s39')
print(list(asample))

print('\nadding labels again')
print('--- adding L1, L2 ----------')
newdataset = add_labels(dataset, labels=['L1', 'L2'])
print(newdataset)

print('\n\n-----++++++ ML data ++++++------')
print('\nReading sample data with labels (as io stream) ------------\n')
data = dataset = dataio.read_data_csv(StringIO(datasets.demo_data2()), has_labels=True)
print(dataset)
print('-- ML data --------------')
print(dataset.ms.data)


