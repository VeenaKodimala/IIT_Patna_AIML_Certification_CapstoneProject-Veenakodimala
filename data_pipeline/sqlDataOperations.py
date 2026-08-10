from BooksDatasetCreation import main
import sqlite3
import pandas as pd

def loadDataset():
    print('Loading the dataset....')
    df = main()
    return df

def createDB(verbose=False):
    print('Establishing the DB connection....')
    dbConnection = sqlite3.connect('BooksDataBase.db')
    cursor = dbConnection.cursor()
    #Checking the type of cursor object.
    if verbose:
        print(type(cursor))

def createDBTables(verbose=False):
    print("Creating the DB Tables...")
    connection = sqlite3.connect('BooksDataBase.db')
    cursor = connection.cursor()
    #Dropping the tables so that the sequence doesn't get affected everytime we run the file.
    cursor.execute('DROP TABLE categories')
    cursor.execute('DROP TABLE books')
    connection.commit()

    categoriesTblCreationQ = '''CREATE TABLE IF NOT EXISTS categories(
    CategoryID INTEGER PRIMARY KEY AUTOINCREMENT,
    CategoryName TEXT NOT NULL UNIQUE
    )'''

    booksCreationTblQ = '''CREATE TABLE IF NOT EXISTS books(
    book_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    title TEXT, 
    price_gbp REAL, 
    price_inr REAL, 
    rating INTEGER, 
    in_stock INTEGER, 
    category_id INTEGER REFERENCES categories(CategoryID)
    )'''

    cursor.execute(categoriesTblCreationQ)
    cursor.execute(booksCreationTblQ)
    connection.commit()

    if verbose:
        cursor.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
        tables=cursor.fetchall()
        for table in tables:
            print(f"ROW: {table}")
    connection.close()    



def insertDataintoDBTables(df,verbose=False):
    print("Inserting data into the created DB tables....")
    connection = sqlite3.connect('BooksDataBase.db')
    cursor = connection.cursor()

    cursor.execute('DELETE FROM categories')

    for cat in df['category'].unique():
        cursor.execute('INSERT INTO categories(CategoryName) VALUES(?)',(cat,))
    connection.commit()    
    cursor.execute('SELECT CategoryName,CategoryID from categories')
    categoryMap = dict(cursor.fetchall())
    if verbose:
        print(f"categoryMap: {categoryMap}")
    df['CategoryID'] = df['category'].map(categoryMap)
    if verbose:
        print(f"DF after adding CategoryID:\n {df.head()}")

    #Convering the datafarame into a list for bulk insert.
    booksData = df[['title','price_gbp','price_inr','Rating','stock','CategoryID']].values.tolist()
    cursor.executemany('INSERT INTO books(title,price_gbp,price_inr,rating,in_stock,category_id) VALUES(?,?,?,?,?,?)', booksData)
    connection.commit()
    if verbose:
        cursor.execute('SELECT * FROM books LIMIT 10')
        rows=cursor.fetchall()
        for row in rows:
            print(f"ROW: {row}")   

def dataAnalysis():
    print('Performing data analysis on data... Start')
    connection = sqlite3.connect('BooksDataBase.db')
    cursor = connection.cursor()

    #SELECT * DEMONSTRATION
    query1='SELECT * FROM CATEGORIES;'
    cursor.execute(query1)
    rows=cursor.fetchall()
    print('''--------------------------QUERY 1: 'SELECT * FROM CATEGORIES;'---------------------------------''')
    print('----------------------------RESULT------------------------------')
    for row in rows:
        print(row)

    #SUB QUERY,ORDER BY DEMONSTRATION
    query2='''SELECT * FROM BOOKS WHERE category_id=(SELECT CATEGORYID FROM CATEGORIES WHERE UPPER(CategoryName)='TRAVEL') ORDER BY book_id DESC;'''
    cursor.execute(query2)
    rows=cursor.fetchall()
    print('''-------QUERY 2: 'SELECT * FROM BOOKS WHERE category_id=(SELECT CATEGORYID FROM CATEGORIES WHERE UPPER(CategoryName)='TRAVEL') ORDER BY book_id DESC;'--------''')
    print('----------------------------RESULT------------------------------')
    for row in rows:
        print(row)    

    #LIMIT DEMONSTRATION
    query3='''SELECT b.CategoryName,a.title,a.rating FROM BOOKS a INNER JOIN CATEGORIES B ON a.category_id=b.CATEGORYID WHERE b.CATEGORYID=3 LIMIT 5;'''
    cursor.execute(query3)
    rows=cursor.fetchall()
    print('''-------QUERY 3: 'SELECT b.CategoryName,a.title,a.rating FROM BOOKS a INNER JOIN CATEGORIES B ON a.category_id=b.CATEGORYID WHERE b.CATEGORYID=3 LIMIT 5;'--------''')
    print('----------------------------RESULT------------------------------')
    for row in rows:
        print(row)    

    #DISTINCT DEMONSTRATION
    query4='''SELECT DISTINCT RATING FROM BOOKS WHERE CATEGORY_ID IN (2,3) ORDER BY RATING;'''
    cursor.execute(query4)
    rows=cursor.fetchall()
    print('''-----------------QUERY 4: 'SELECT DISTINCT RATING FROM BOOKS WHERE CATEGORY_ID IN (2,3) ORDER BY RATING;'-----------------''')
    print('----------------------------RESULT------------------------------')
    for row in rows:
        print(row)          

    #DISTINCT DEMONSTRATION
    query5='''SELECT * FROM BOOKS WHERE PRICE_INR BETWEEN 4000 AND 5000 LIMIT 5;'''
    cursor.execute(query5)
    rows=cursor.fetchall()
    print('''-----------------QUERY 5: 'SELECT * FROM BOOKS WHERE PRICE_INR BETWEEN 2000 AND 5000;'-----------------''')
    print('----------------------------RESULT------------------------------')
    for row in rows:
        print(row)     

    #DISTINCT DEMONSTRATION
    query6='''SELECT category_id,count(*) FROM BOOKS a INNER JOIN CATEGORIES B ON a.category_id=b.CATEGORYID GROUP BY a.category_id;'''
    cursor.execute(query6)
    rows=cursor.fetchall()
    print('''-----------------QUERY 5: 'SELECT count(*) FROM BOOKS a INNER JOIN CATEGORIES B ON a.category_id=b.CATEGORYID GROUP BY a.category_id;'-----------------''')
    print('----------------------------RESULT------------------------------')
    for row in rows:
        print(row)             

def dbAnalysisUsingpandas(verbose=False):
    print('Using pandas to perform data analysis...')
    connection = sqlite3.connect('BooksDataBase.db')
    cursor = connection.cursor()

    pdQuery1 = "SELECT * FROM CATEGORIES;"
    pdQuery2 = "SELECT * FROM BOOKS;"

    categoriesDF = pd.read_sql(pdQuery1,connection)
    if verbose:
        print(categoriesDF)

    booksDF = pd.read_sql(pdQuery2,connection)
    if verbose:
        print(booksDF.head(5))


    query5Result = pd.merge(left=booksDF,right=categoriesDF,how='inner',left_on=['category_id'],right_on=['CategoryID']).groupby(by=['CategoryID']).agg({'title':'count'})



    print(f"Reproducing the QUERY 5: 'SELECT count(*) FROM BOOKS a INNER JOIN CATEGORIES B ON a.category_id=b.CATEGORYID GROUP BY a.category_id;' result: \n{query5Result}")



#------------------------------------------ENTRY POINT OF THE CODE-----------------------------------------------------------------


def entryPoint(verbose=False):
    df = loadDataset()    
    if verbose:
        print(f"Loaded Dataset successfully!!!....")
        print(df.head(5))
    createDB()
    createDBTables()
    insertDataintoDBTables(df)
    dataAnalysis()

    dbAnalysisUsingpandas()


if __name__ == '__main__':
    print("Inside the entry of sqlDataOperations...")
    entryPoint()    

