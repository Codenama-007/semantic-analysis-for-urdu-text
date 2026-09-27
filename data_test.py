import pandas as pd

# laoding the dataset
data = pd.read_csv("Complete_Quran_data.csv")


cats = []

for cat in data['Category']:
    cats.append(cat)
    
    count += 1
    
    # if count == 15:
    #     break
    
    # else:
    #     continue
    
for cat in cats:
    print(cat)
    print("="*25)
    
    
print(data["Category"].value_counts())