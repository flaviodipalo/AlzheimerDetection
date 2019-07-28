import pandas as pd
import pickle

df = pd.read_csv('data/anagraphic_modded.csv', sep=';')
print(list(df))
imputed_dict_list = []

visit_dict = {'visit2':1,'visit3':2,'visit4':3,'visit5':4,'visit6':5,'visit7':6}

print('Preprocessing Anagraphic Files...')
for index, row in df.iterrows():
    id = str(row['id']).zfill(3)
    entry_age = int(row['entryage'])
    initial_date = int(row['idate'].split('-')[-1])

    patient_dict = {
        'id': id+'-0',
        'age': entry_age,
        'sex': row['sex'],
        'race': row['race'],
        'education': row['educ']
    }
    imputed_dict_list.append(patient_dict)

    for key in visit_dict.keys():
        if type(row[key]) == str:
            patient_dict = {
                'id': id + '-' + str(visit_dict[key]),
                'age': entry_age + int(row[key].split('-')[-1]) - initial_date,
                'sex': row['sex'],
                'race': row['race'],
                'education': row['educ']
            }
            imputed_dict_list.append(patient_dict)

final_dataframe = pd.DataFrame(imputed_dict_list)

print(final_dataframe.head())

with open('data/anagraphic_dataframe.pickle', 'wb') as f:
    pickle.dump(final_dataframe, f)

