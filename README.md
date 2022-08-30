# Loan-return-prediction-using-Artificial-neural-networks-ANN-
Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), we are building an model that can predict whether or not a borrower will pay back their loan. 


•	Data: 
We are using a subset of the LendingClub DataSet  from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club


•	Company Info: 
 LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. 
It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.


•	Goal:  
Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), we are building an model that can predict whether or not a borrower will pay back their loan. 
This way in the future when we get a new potential customer we can assess whether or not they are likely to pay back the loan. 
We have to keep in mind classification metrics when evaluating the performance of our model.
The "loan_status" column contains our label.


•	Dataset Information:
Here is the information on this particular data set:


•	Model Description:


	We started by doing feature engineering on the dataset:

o	We attempted to predict loan_status, so we created a count plot.
o	Histogram of the loan_amnt column to see the distribution.
o	Explored correlation between the continuous feature variables
o	Installment had nearly perfect correlation with loan_amnt  so, we explored that further with scatterplot.
o	We visualized subgrades, It looked like F and G subgrades don't get paid back often. We isolated those and recreated the countplot for those subgrades.
o	Then we started doing pre-processing of our dataset.
o	Removed the emp_title column, dropped the emp_length column and dropped the title column because of our inability to feed these in the model creation.
o	With some visualization and analysis, with help of a function, we filled in the missing mort_acc values based on their total_acc value.
o	In revol_util  column, dropped the rows that are missing those values in those columns with dropna().
o	Extracted Zip code from the address column and year from earliest_cr_line, then treated it as a categorical features.
o	For categorical features, converted them into object type and then created dummy variables for them.

	Then we did model building:
o	Train Test Split of the feature and columns.
o	Normalized the data.
o	Created the Model that goes 78 --> 39 --> 19--> 1 output neuron, activation as ‘relu’, ‘sigmoid’ for output, loss='binary_crossentropy' and optimizer='adam'.

	Model Evaluation:
o	Classification report was printed:   

	Hence our model had an accuracy of 0.89 in predicting whether a person whom we are giving loan will return the loan or not, given the features about the person in accordance to the historical data.
