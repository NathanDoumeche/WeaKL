# WeakL applied to time series

This project aims at illustring the results of the three use cases in the paper _Forecasting time series with constraint_ (Nathan Doumèche, Francis Bach, Eloi Bedek, Gérard Biau, Claire Boyer, and Yannig Goude).

## Install
To install packages used to generate the results in the paper, create a virtual environment using python 3.10.16. Then, run the following command in the terminal.
```bash
pip install -r requirements.txt
```
## Use case 1: the IEEE DataPort competition on day-ahead electricity load forecasting
Go to the corresponding directory and unzip the file _data\_corr.csv.zip_. There are one notebook and a python script. The _WeakL.ipynb' notebook generate the forecast of the WeakL model, that is a direct translation of the GAM model equation into a WeakL framework.

1) Unzip the _dataset_national.csv.zip_ file. 

2) Create a virtual environment with Python 3.9 and install the packages of _requirements.txt_.


