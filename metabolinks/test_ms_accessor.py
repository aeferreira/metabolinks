from six import StringIO
import pandas as pd
import metabolinks.dataio as dataio
import metabolinks.demodata as demodata
import metabolinks.transformations as transformations


print('\nReading sample data (as io stream) ------------\n')
dataset = dataio.read_data_csv(StringIO(demodata.demo_data1()))
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
data = dataset = dataio.read_data_csv(StringIO(demodata.demo_data2()), has_labels=True)
print(dataset)
print('-- info --------------')
print(dataset.ms.info())
print('-- global info---------')
print(dataset.ms.info(all_data=True))
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
print(dataset.ms.labels)

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
dataset = dataio.read_data_csv(StringIO(demodata.demo_data2()), has_labels=True)
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
newdataset = dataset_unlabeled.ums.add_labels(labels=['L1', 'L2'])
print(newdataset)
