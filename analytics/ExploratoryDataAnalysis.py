import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def biVariantAnalysis(df):
    print("--------------------SURVIVAL RATE BASED ON SEX COLUMN-----------------------")
    uniqueueSex= df['sex'].unique()
    for sex in uniqueueSex:
        survivalrate = (df.loc[df['sex'] == sex,'survived'].mean())
        print(f"SEX:: {sex}::: {survivalrate:.2%}")

    print("--------------------SURVIVAL RATE BASED ON PCLASS COLUMN-----------------------")
    uniquePClass = df['pclass'].unique()
    for pclass in uniquePClass:
        survivalrate = (df.loc[df['pclass'] == pclass,'survived'].mean())
        print(f"PCLASS::: {pclass}::: {survivalrate:.2%}")

    print("--------------------SURVIVAL RATE BASED ON PCLASS AND SEX COLUMN-----------------------")
    uniquePClass = df['pclass'].unique()
    uniqueueSex = df['sex'].unique()

    for pclass in uniquePClass:
        for sex in uniqueueSex:
            survivalrate = (df.loc[((df['pclass'] == pclass) & (df['sex'] == sex)),'survived'].mean())
            print(f"PCLASS::: {pclass}, SEX::: {sex}::: {survivalrate:.2%}")      

    cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
    corr = df[cols].corr()
    print(f"Correlation Matrix: \n{corr}")

    plt.figure(figsize=(8, 6))
    sb.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')


    # ── Top 2 strongest off-diagonal correlations ─────────────────────────────────

    mask_upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = (corr.where(mask_upper)
                 .stack()
                 .reset_index()
                 .rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', 0: 'correlation'}))
    pairs['abs_corr'] = pairs['correlation'].abs()
    top2 = pairs.nlargest(2, 'abs_corr')
    print("\n-----------------------Top 2 Strongest Correlations------------------------")
    print(top2[['feature_1', 'feature_2', 'correlation']].to_string(index=False))

def multiVariantAnalysis(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Titanic Survival — Multivariate Data Story', fontsize=16, fontweight='bold')

    # Chart 1 — Survival Rate by Class (Bar)
    sb.barplot(data=df, x='pclass', y='survived', palette='Blues_d', errorbar='ci', ax=axes[0, 0])
    axes[0, 0].set_title('Survival Rate by Passenger Class')
    axes[0, 0].set_xlabel('Passenger Class')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].set_xticks([0, 1, 2])
    axes[0, 0].set_xticklabels(['1st', '2nd', '3rd'])
    
    # Chart 2 — Age Distribution by Survival (Box)
    sb.boxplot(data=df, x='survived', y='age', palette={'0': '#E07070', '1': '#70A8E0'}, ax=axes[0, 1])
    axes[0, 1].set_title('Age Distribution by Survival')
    axes[0, 1].set_xlabel('Survived (0=No, 1=Yes)')
    axes[0, 1].set_ylabel('Age')
    
    # Chart 3 — Fare vs Age Scatter
    colors = df['survived'].map({0: '#E07070', 1: '#70A8E0'})
    axes[0, 2].scatter(df['age'], df['fare'], c=colors, alpha=0.5, edgecolors='white', linewidths=0.3)
    axes[0, 2].set_title('Fare vs Age by Survival')
    axes[0, 2].set_xlabel('Age')
    axes[0, 2].set_ylabel('Fare')
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E07070', markersize=8, label='No'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#70A8E0', markersize=8, label='Yes')
    ]
    axes[0, 2].legend(handles=handles, title='Survived')
    
    # Chart 4 — Correlation Heatmap
    corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
    corr = df[corr_cols].corr()
    sb.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation Heatmap')
    
    # Chart 5 — Survival Rate by Sex (Bar)
    sb.barplot(data=df, x='sex', y='survived', palette={'male': '#70A8E0', 'female': '#E07070'},
                errorbar='ci', ax=axes[1, 1])
    axes[1, 1].set_title('Survival Rate by Sex')
    axes[1, 1].set_xlabel('Sex')
    axes[1, 1].set_ylabel('Survival Rate')
    
    # Chart 6 — Survival by Class & Sex (Grouped Bar)
    sb.barplot(data=df, x='pclass', y='survived', hue='sex',
                palette={'male': '#70A8E0', 'female': '#E07070'}, ax=axes[1, 2])
    axes[1, 2].set_title('Survival by Class & Sex')
    axes[1, 2].set_xlabel('Passenger Class')
    axes[1, 2].set_ylabel('Survival Rate')
    
    plt.tight_layout()
    plt.savefig('titanic_full_story.png', dpi=150)

def manualStandardization(df):
    for col in ['age', 'fare']:
        mean = df[col].mean()
        std  = df[col].std()
        df[f'{col}_scaled'] = (df[col] - mean) / std

# ── Before / After summary ────────────────────────────────────────────────────
    print("=== Before Standardization ===")
    print(df[['age', 'fare']].agg(['mean', 'std']).round(4))
    
    print("\n=== After Standardization ===")
    print(df[['age_scaled', 'fare_scaled']].agg(['mean', 'std']).round(4))


def entryPoint():
    titanicDataset = loadDataset()
    failSafeDataset(titanicDataset)
    titanicDataset=handlingMissingValues(titanicDataset)
    print(f"--------------Infor of Dataset after Handling Missing Values--------------------")
    print(titanicDataset.info())
    univariantAnalysis(titanicDataset)
    biVariantAnalysis(titanicDataset)
    multiVariantAnalysis(titanicDataset)
    manualStandardization(titanicDataset)
    



if __name__ == '__main__':
    entryPoint()