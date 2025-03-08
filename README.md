# import-wellsfargo-pdf-statement
Python project to import statement transactions into Pandas and write to a CSV, Excel, JSON or Parquet file.

The code can process individual or multiple files and folders. Optionally write them into multiple or single files.

This code reads the statement summary information and statement activity.

The summary information include the period start and end. It also include the beginning and ending balance as well as the total credits and debits.

The summary information is used to validate the statement activity and ensures all the values were correctly inported.

This has been tested with statements from 2019 up to 2025.

Based on the code from [Julian Kingman](https://github.com/JulianKingman/wells-fargo-statement-to-csv).
