import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL="https://books.toscrape.com/"

def getCategories(verbose=False,limit=3) -> list:
    print('Getting the list of categories...')
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.content,'html.parser')
    categoryLinks = soup.select('ul.nav-list > li > ul > li > a')
    categories=[]
    if verbose:
        print(categoryLinks)

    #print(categoryLinks)

    for link in categoryLinks:
        subLink = link['href']
        catName=link.text.strip()

        categories.append({catName:BASE_URL+subLink})
        if verbose:
            print(f"categories:\n{categories}") 
    return categories[:limit]   

def getAllBooks(categoryDets,verbose=False):
    print("Getting the books for the categories fetched...")
    allBooks=[]
    for categorySet in categoryDets:
        catName=list(categorySet.keys())[0]
        catLink=list(categorySet.values())[0]
        if verbose:
            print(f"Category: {catName}")
            print(f"Category Link: {catLink}")
        catBooks = getBooksPerCategory(catLink,catName)   
        allBooks.extend(catBooks)
        if verbose:
            print(f"allBooks: {allBooks}")
            print(f"No.of Books fetched : {len(allBooks)}")
    return allBooks        



def getBooksPerCategory(categoryLink,categoryName,verbose=False):    
    catBooks=[]    
    #Here, both will work, but we need to give the shorter path, as long as the element is idetified uniquely.

    while categoryLink:
       response = requests.get(categoryLink)
       soup = BeautifulSoup(response.content,'html.parser')
       books = soup.select('article.product_pod')
       #books = soup.select('div > ol.row > li > article.product_pod > h3 > a')
       for book in books:
         if verbose:
               print(book)
         title = book.h3.a['title']
         price = book.select_one('p.price_color').text.strip()
         starRating = book.select_one('p.star-rating')["class"][1]
         stock = book.select_one('p.availability').text.strip()
         stock = book.select_one('p.availability').text.strip() if stock else 'Out of Stock'
         if verbose:
            print(stock)   
         catBooks.append({'category':categoryName,'title':title,'price':price,'Rating':starRating,'stock':stock})
       nextBtn = soup.select_one('li.next > a')
       if verbose:
           print(f"nextBtn: {nextBtn}")
       categoryLink = (categoryLink.rsplit("/",1)[0])+"/"+nextBtn['href'] if nextBtn else None
       if verbose:
           print(f"New categoryLink: {categoryLink}")

    return catBooks   

def cleanDataset(booksDataset,verbose=False):
    print('Cleaning the dataset internally...')
    #print(f"Dataset received: {booksDataset.head(5)}")
    booksDataset['price_gbp'] = (booksDataset['price']).str.replace('£','').astype(float)
    if verbose:
        print(f"Dataset after replacing currency symbol in price column: \n{(booksDataset.head(5))}")
        print(booksDataset.info())
        print(booksDataset['Rating'].unique())
    booksDataset['Rating'] = booksDataset['Rating'].map({'One':1,'Two':2,'Three':3,'Four':4,'Five':5}).astype(int)    
    if verbose:
        print(f"Dataset after cleaning rating column: \n{(booksDataset.head(5))}")
        print(booksDataset.info())
    #Label encoding - Stock column
    booksDataset['stock'] = booksDataset['stock'] == 'In stock'
    if verbose:
        print(f"Dataset after cleaning rating column: \n{(booksDataset.head(5))}")
        print(booksDataset.info())
   
    booksDataset['price_inr'] = ((booksDataset['price_gbp']) * 105.50).round(2)
    if verbose:
        print(f"Dataset after creating price_inr column: \n{(booksDataset.head(5))}")
        print(booksDataset.info())

    #Dropping unnecessary columns: price
    booksDataset = booksDataset.drop(columns=['price'],errors='ignore')
    if verbose:
        print(f"Dataset after creating price_inr column: \n{(booksDataset.head(5))}")
        print(booksDataset.info())

    return booksDataset

        



def main(verbose=False):
    categoryDets = getCategories()
    if verbose:
        print(f"Category details received from getCategories method: {categoryDets}")
    allBooks = getAllBooks(categoryDets=categoryDets)
    if verbose:
        print(f"Total No.of Books fetched from source: {len(allBooks)}")
        print(f"Books fetched from the source: \n{allBooks}")

    booksDataset = pd.DataFrame(allBooks)
    if verbose:
        print(f"Converted to dataframe: \n{booksDataset}")
    cleanedDataset=cleanDataset(booksDataset)    
    if verbose:
        print(f"Dataset after cleaning activity: \n{cleanedDataset.info()}")
    return cleanedDataset









if __name__ == '__main__':
    main()    


