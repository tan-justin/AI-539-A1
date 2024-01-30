Machine Learning Challenges


Description: 

This program is designed to showcase the different methods of handling missing data. The methods utilized in the code are as follows:

A: Abstention
B: Majority Inference
C: Feature Exclusion
D-F: Imputation utilizing 3 different strategies

The csv files are sourced from the following website: https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/CFHTLens/query.html

The program will first generate a data profile of the features found in cfhtlens.csv to understand the missingness of the data. It will then perform a train-test split of the data. In train_eval.py, the data is split such that the items in the training dataset do not contain any missing values while the test dataset contains a mix of items with missing values and no missing values. In extra_credit.py, both the training dataset and test dataset contain a random mix of items containing missing values and no missing values. Methods A to F are performed on the testing dataset for train_eval.py. Methods C to F are performed on both the training and testing dataset for extra_credit.py. The program then uses a random forest classifier to classify the items in the test dataset. The classification accuracy is stored in a dictionary for the predicited classification of the test dataset after each method of handling missing value to better understand how effective each strategy is. 


Instructions:

To run the program, clone the repository and open the directory. Run the program using python3 main.py


Packages required: 

Numpy, Pandas, Scikit-Learn, ReportLab, MatPlotLib

Links to install these packages:

Numpy: https://numpy.org/install/

Pandas: https://pandas.pydata.org/docs/getting_started/install.html

Scikit-Learn: https://scikit-learn.org/stable/install.html

MatPlotLib: https://matplotlib.org/stable/users/installing/index.html

ReportLab: https://pypi.org/project/reportlab/


Credits: 

This was one of the homework assignment associated with the Machine Learning Challenges (Winter 2024 iteration) course at Oregon State University. All credits belong to the course developer, Dr. Kiri Wagstaff and the lecturer-in-charge, Prof. Rebecca Hutchinson. All code is written by myself solely for the purpose of the assignment. For more information on the course developer and lecturer-in-charge:

Dr. Kiri Wagstaff: https://www.wkiri.com/

Prof. Rebecca Hutchinson: https://hutchinson-lab.github.io/


Use: 

The code shall be used for personal educational purposes only. Students of current (Winter 2024) and future iterations of Machine Learning Challenges at Oregon State University may not use any code in this repo for this assignment should this assignment be assigned. If any Oregon State University student is found to have plagarized any code in this repo, the author of the repository cannot be held responsible for the incident of plagarism. The author promises to cooperate in any investigations regarding plagarism pertaining to this repo if required. If any of the code in this repo is reused for strictly personal projects, please credit this repository. 
