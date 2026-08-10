# Data Pipeline

This folder contains the data pipeline for scraping, cleaning, storing, and analysing a books dataset using web scraping, pandas, and SQLite.

## Files

### `BooksDatasetCreation.py`
Scrapes book data from [books.toscrape.com](https://books.toscrape.com/) and returns a cleaned pandas DataFrame.

**Key functions:**

| Function | Description |
|---|---|
| `getCategories(verbose, limit)` | Fetches book category names and URLs from the homepage. Default `limit=3`. |
| `getAllBooks(categoryDets, verbose)` | Iterates over categories and collects all book records. |
| `getBooksPerCategory(categoryLink, categoryName, verbose)` | Scrapes all paginated books within a single category. |
| `cleanDataset(booksDataset, verbose)` | Cleans and transforms raw data: strips currency symbol, maps star ratings to integers (1–5), encodes stock as boolean, adds `price_inr` column (GBP × 105.50), drops the original `price` column. |
| `main(verbose)` | Orchestrates the full scrape and returns the cleaned DataFrame. |

**Output DataFrame columns:** `category`, `title`, `Rating`, `stock`, `price_gbp`, `price_inr`

---

### `sqlDataOperations.py`
Loads the scraped dataset into a SQLite database and performs SQL and pandas-based analysis.

**Key functions:**

| Function | Description |
|---|---|
| `loadDataset()` | Calls `BooksDatasetCreation.main()` and returns the cleaned DataFrame. |
| `createDB(verbose)` | Opens (or creates) `BooksDataBase.db` and establishes a connection. |
| `createDBTables(verbose)` | Drops and recreates the `categories` and `books` tables. |
| `insertDataintoDBTables(df, verbose)` | Inserts category and book records into the database using bulk insert. |
| `dataAnalysis()` | Runs six SQL queries demonstrating SELECT, subquery, ORDER BY, JOIN, LIMIT, DISTINCT, BETWEEN, and GROUP BY. |
| `dbAnalysisUsingpandas(verbose)` | Replicates a GROUP BY aggregation using `pd.read_sql` and `DataFrame.merge`. |
| `entryPoint(verbose)` | Main orchestrator: loads data → creates DB → creates tables → inserts data → runs analysis. |

**Database:** `BooksDataBase.db` (SQLite, created in the working directory)

**Tables:**

- `categories(CategoryID, CategoryName)`
- `books(book_id, title, price_gbp, price_inr, rating, in_stock, category_id)`

---

## How to Run

```bash
cd data_pipeline
python sqlDataOperations.py
```

This executes the full pipeline: scrapes books, stores them in SQLite, and prints SQL and pandas analysis results.

## Dependencies

- `requests`
- `beautifulsoup4`
- `pandas`
- `sqlite3` (standard library)
