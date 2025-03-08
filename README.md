# Stock Price Trend Classification with Random Forest

## Overview 
Many individuals aim to invest their money in stocks and take advantage of the seemingly quick, but also risky way of making money. The movement of stock prices are impacted by so many factors such as crowd sentiment, world news, macroeconomic movement, and the intrinsic value of a company’s assets. Stock analysis is divided into two main categories: fundamental and technical analysis. Fundamental analysis focuses on the broader picture; it places an emphasis on a company’s financial records and market capitalization, while technical analysis focuses on creating an almost scientific method of stock prediction that’s based on patterns, trends, and momentum of a stock’s price. Keeping track of all the indicators and signals dictated in the technical analysis of a stock can be complex and time-consuming to do by hand, which is why machine learning techniques can be used to aid this process. Techniques such as Random Forest can essentially capture all these indicators and signals as features and classify the stock price as going up or down. This kind of classification offers investors a buy or sell signal from the technical perspective of stock prediction, which they can verify with real-world sentiments to help make decisions about entry and exit points for investments. 

The dataset used to train and test the model is a dataset with the historical price data of the S&P500 to capture patterns consistent with the top 500 traded stocks in the United States.

This particular rendition of a stock trend classifier labels the training data with a 5-day lagged close price to simulate conditions for the moderately active home investor. If P(i) represents the most recent close on the i-th day then:

P(i) - P([i-5]) > 0 indicates that the price has increased within the past 5 days. <br>
P(i) - P([i-5]) <= 0 indicates that the price has decreased within the past 5 days.

Using this 5-day lag also helps the model filter out noise in the data that occurred from a 1-day lag close price, and the model performed significantly better with the increased lag, improving the accuracy by about 18%.

## Why Random Forest? (Advantages and Disadvantages)
Random Forest Classification is an intuitive choice for stock bullish/bearish analysis because the strategies that many traders use can be condensed to a more qualitative decision tree. When analysts make decisions about a security, they filter through different signals and indicators, making often binary decisions about which technical or market conditions indicate a bullish or bearish move for a stock, before coming to a final conclusion (buy or sell). For example, a trader might first look at the RSI indicator and see that it has dipped below the 30-line indicating that it is oversold. They may also see that the ADX is greater than 15 and DMI+ is above the DMI- line, indicating that there is an uptrend forming. After looking at several other indicators, they will come to a conclusion of a certain confidence that there is a strong chance that the stock will go up. Decision trees in machine learning capture these decision points in a quantitative way, determining the best way to split each decision. If we go one step further, we can imagine that evaluating the opinion of 100 traders each with their own conclusion would result in a more reliable estimation of the "correct" classification of a stock. The intent behind Random Forest is exactly this, where 100 different decision trees each produce an outcome and the final prediction is determined on a majority vote basis. 

Notice how these decisions can be represented in a tree. [insert tree]

Random Forest classification captures this same process with more mathematical derivations of decision making based on maintaining node purity. Each point of decision in a decision tree aims to create the largest disparity between labels. For example, there are 300 "bullish" data points and 20 "bearish" data points with RSI less than or equal to 40 vs. 200 "bearish" data points and 50 "bullish" data points with RSI greater than 40. This is identified as a determining split level because one side of the split has mostly bullish data points while the other side has mostly bearish data points. There is clearly some information being gained with this split.

Compare this to RSI less than 60 and greater than 60 which may separate 300 bullish data points and 100 bearish points from 100 bearish data points and 50 bullish data points. With this split, there is less purity in each side of the split -- a more heterogenous mixture of both classes in each node. This ambiguity suggests that the RSI being less than or greater than 60 isn't a deciding factor in classifying a data point as bullish or bearish. 

This idea is the essence of decision trees and the deterministic gini index, which is a measure of impurity (non-homogeneity of nodes). A decision tree aims to minimize the gini index at each decision point until it reaches the maximum depth which leaves a handful of samples in the leaf nodes. 

Another important advantage of using Random Forest over just one decision tree besides having more than one "opinion" on a classification is how each tree is independent to one another because of feature subsetting and bagging. The decorrelation of trees is imperative in terms of avoid overfitting -- the model memorizes the noise of the data. 

### Feature Subsetting
Each tree in the Random Forest gets the full set of features available, but if it just went by the best feature to split by at every node, we would end up with a forest filled with identical decision trees. This isn't very useful! To address this, Random Forest takes the best feature among a random subset of features while splitting a node. For example, if we had features {RSI, MACD, ADX, Volume} and the sample size of the subset was 2, at split 1, we might get the subset {RSI, ADX} and determine that the best feature to split by is the RSI. At the subsequent split, we take another subset of the original set of features with replacement, {RSI, Volume} and repeat the process. The number of features sampled at each node is denoted by "m" and usually set to the square root of the total number of features n. However, this is a hyperparameter that can be tuned to see what best fits each model. 

### Bagging 
Bagging is used to further decorrelate each tree by taking a random bootstrapped sample of data points to train each tree. Bootstrapping means that the original set of data points is sampled with replacement so each tree may be trained on repeated points. Random Forest avoid overfitting through this means of shuffling the data. 

### Disadvantages of Random Forest
Random Forest is computationally expensive. The more estimators (number of trees), the longer it takes for the algorithm to run, especially if each tree maintains its max depth. This can potentially be avoided by reducing how deep the tree can go and adjusting the number of estimators, but there's usually a tradeoff between computation and accuracy here. 

Random Forest is also black box. It's not feasible to see the different branches of each of 100s of decision trees, so a lot of this information isn't shown in detail while working with the Random Forest model. This makes it harder to tune and adjust the model. However, cross-validation is an efficient way to address this issue. 

## Dataset
The dataset used was a Kaggle Dataset with data on open, high, low, close, volume, and adjusted close prices for many stocks. The AAPL dataset was used for this project: www.kaggle.com/datasets/jacksoncrow/stock-market-dataset. Volatility index data was sourced from Python's YFinance (Yahoo Finance) library.

## Methodology
In this exploration of the applications of machine learning, I will focus on classifying the direction (up or down) of stocks, focusing on the stock price of Apple (AAPL). AAPL is known for its stability in the markets, which makes it a beneficial choice in this study in terms of reducing noise. The two classes are uptrend determined by 1 and downtrend determined by 0. These response variables are calculated by using a 5-day lag beteween close-prices: 
Price_Difference = Price(i) - Price(i - 5) where i is the day in consideration. If Price_Difference is positive, this is counted as an uptrend; if negative, this is marked as a downtrend. Essentially, the model aims to see the general trend of a price over the next 5 days. This decision was made to reduce the noise that occurs when considering the price trend after just 1 day and smooth out the prices so the algorithm could perform better. 

The features in consideration are several different indicators such as RSI, MACD, Williams %R, ADX, and Bollinger Bands. The main algorithm used was Random Forest, which is one of the industry standards for classification in terms of accuracy and prevention of overfitting. This project makes use of many different data science skills such as wrangling and pre-processing, feature engineering, data exploration through plots and correlation matrices, and model analysis using metrics such as accuracy, precision, specificity, sensitivity, and F1 score.  

## Results
One of the evaluation methods used is the Out-Of-Bag Error (OOB Error), which allows Random Forests to perform its own validation while training the model. Approximately, each resampled dataset will contain 67% of the original data, so about 33% of the original dataset hasn’t been seen by each tree. This allows for that 33% of “out-of-bag” datapoints to be used as a validation set for each tree. The OOB error represents how many of the out-of-bag points were misclassified. This is important because it opens the door for cross-validation of tuning parameters based on the OOB Error. Below is the OOB error result from training the Random Forest model. The calculated OOB-Error is 15.28% indicating an 85% accuracy. 

![image](https://github.com/user-attachments/assets/f6ca01da-e4d1-46bf-8848-ddd7411f62e9)

Another validation set was used with more unseen data to offer more confidence in the model. The 'Positive Class' is the downtrend class (0). The results of the accuracy, sensitivity, specificity, precision (Positive Predictive Value), and F1 score (balanced accuracy) are also shown as well as the related confusion matrix. Comparing the accuracy to the base-line no-information rate, the accuracy with random guessing, there is a significant improvement, from 56.83% (base-line) to 81.93% with the model. The accuracy of the validation set compared to the accuracy determined by the OOB-error has a 3% difference (81.93% to 85%), which most likely means the model isn't overfitting the data. Additionally, the PPV shows that the model is better at classifying uptrends than downtrends. The detection rate is also really low compared to the rest of the metrics, showing that there needs to be improvement in terms of correctly predicting downtrends. 

![image](https://github.com/user-attachments/assets/55b6f494-26e2-4a43-a518-9b3e2caca3cb)

Feature importance showed that Williams %R, RSI, and the ADX & directional movement -- trend strength -- index features were the most important features in terms of contribution to accuracy and decrease of gini index when constructing the random forest model. 

![image](https://github.com/user-attachments/assets/a159e28e-db76-4399-bca0-32b1f5cb320f)
