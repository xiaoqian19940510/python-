import pandas as pd
#data process
train1=pd.read_csv('taijing/df_affinity_train.csv')
train2=pd.read_csv('taijing/df_molecule.csv')
test=pd.read_csv('taijing/df_affinity_test_toBePredicted.csv')

train1=pd.DataFrame(train1)
train2=pd.DataFrame(train2)
test=pd.DataFrame(test)

test.columns = ['Protein_ID','Molecule_ID','Ki']
test['Ki']=0
del test['Ki']
# del test['Ki']
train1.columns = ['Protein_ID','Molecule_ID','Ki']
# train1.dropna(inplace=True)
train1.fillna(0.0,inplace=True)
train2.columns = ['Molecule_ID','Fingerprint','cyp_3a4','cyp_2c9','cyp_2d6','ames_toxicity','fathead_minnow_toxicity','tetrahymena_pyriformis_toxicity','honey_bee','cell_permeability','logP','renal_organic_cation_transporter','CLtotal','hia','biodegradation','Vdd','p_glycoprotein_inhibition','NOAEL','solubility','bbb']
# train2.dropna(inplace=True)
del train2['Fingerprint']
train2.fillna(0.0,inplace=True)
test.fillna(0.0,inplace=True)
train1.fillna(0.0,inplace=True)
# train2.to_csv('taijing/df_molecule_drop.csv')
# test_fianll=test.concat(train2, keys=['Molecule_ID'])
test_fianll=pd.merge(test,train2)
train_finall=pd.merge(train1,train2)
test_fianll.fillna(0.0,inplace=True)
train_finall.fillna(0.0,inplace=True)
# print(train.head(6))
train_finall.to_csv('taijing/df_affinity_train_combine.csv')
test_fianll.to_csv('taijing/df_affinity_test_combine.csv')
