import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

def loadDataset():
    titanicDF = sb.load_dataset('titanic')
    print('------------------Dataset Loaded!!!----------------')
    print(titanicDF.head(5))
    print('------------------Info of Dataset----------------')
    print(titanicDF.info())
    print('------------------Description of Dataset----------------')
    print(titanicDF.describe())
    print('------------------Shape of Dataset----------------')
    print(titanicDF.shape)

    computeMissingReport(titanicDF)
    return titanicDF

def failSafeDataset(df):
    df.to_csv("titanic.csv", index=False)

def handlingMissingValues(titanicDataset,verbose=False):
    titanicDataset = titanicDataset.dropna(subset=['embarked'])    
    '''
    Imputing the age column as follows:
    1. Filling the age column where alive=No, with mode obtained from rows when alive=No
    2. Filling the age column where alive=Yes, with mode obtained from rows when alive=Yes
    '''
    # Get mode of age where alive = 'No'
    mode_age_dead = titanicDataset[titanicDataset['alive'] == 'no']['age'].mode()[0]
    mode_age_alive = titanicDataset[titanicDataset['alive'] == 'yes']['age'].mode()[0]

    if verbose:
        print(f"mode_age_dead: {mode_age_dead}")
        print(f"mode_age_alive: {mode_age_alive}")

    # Fill nulls only where alive = 'No', with mode_age_dead
    titanicDataset.loc[(titanicDataset['age'].isnull()) & (titanicDataset['alive'] == 'no'), 'age'] = mode_age_dead
    titanicDataset.loc[(titanicDataset['age'].isnull()) & (titanicDataset['alive'] == 'yes'), 'age'] = mode_age_alive

    #Filling the missing values of 'deck' column with new category 'missing'
    titanicDataset['deck'] = titanicDataset['deck'].cat.add_categories('Missing')
    titanicDataset['deck'] = titanicDataset['deck'].fillna('Missing')

    computeMissingReport(titanicDataset)

    return titanicDataset

    
def univariantAnalysis(df):
    fig,ax=plt.subplots(1,2,figsize=(20,15))
    sb.boxplot(y=df['age'],color='violet',ax=ax[0])
    ax[0].set_ylabel('Age')
    ax[0].set_title('Distribusion of Age')

    sb.boxplot(y=df['fare'],color='green',ax=ax[1])
    ax[1].set_ylabel('Fare')
    ax[1].set_title('Distribusion of Fare')
    plt.savefig('BoxPlotDistribution-UnivariantAnalysis.png')

    for col in ['age','fare']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) |(df[col] > upper)]
        print(f"{col}: {len(outliers)} outliers  (lower={lower:.2f}, upper={upper:.2f})")




    


def computeMissingReport(titanicDataset):
    print('-----------------Missing Values Report----------------------')
    missingDF = pd.DataFrame(
                {
                    "missing count":titanicDataset.isnull().sum(),
                    "missing percentage":(titanicDataset.isnull().sum() / len(titanicDataset) * 100).round(2)
                }
            )
    print(missingDF[missingDF['missing count'] > 0])    

    



def entryPoint():
    titanicDataset = loadDataset()
    failSafeDataset(titanicDataset)
    titanicDataset=handlingMissingValues(titanicDataset)
    print(f"--------------Infor of Dataset after Handling Missing Values--------------------")
    print(titanicDataset.info())
    univariantAnalysis(titanicDataset)



if __name__ == '__main__':
    entryPoint()