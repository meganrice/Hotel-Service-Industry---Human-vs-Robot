# Hotel Service Industry - Human vs Robot
Text Mining &amp; Social Media Analytics Final Project

#### Background
For any popular product or service, reviews are abundant. This abundance of reviews rings especially true for the fiercely competitive hotel industry. With an overload of feedback, it is difficult for hotel managers to keep track of and manage customer opinions. To counter this problem, we propose a supervised machine learning approach using unigram features (Countvectorizer and TF-IDF) to create polarity classification of hotel reviews. We will test the performance and compare multiple machine learning, deep learning, and sentiment scoring models to recommend the most effective model for building a sentiment monitoring system for hotel managers. With the help of our best-performing model, this screening system will help managers automatically detect, in real-time, the highest number of true positives and negatives based on review ratings (high or low).

The intense competition in the hotel industry explains why, in recent years, they have been experimenting with the implementation of robots and AI. The hotel industry is always looking for the next innovation to maximize customer satisfaction cost-effectively. It may sound like a science fiction experience to arrive at a hotel in a driverless taxi, check-in at a reception desk staffed by an android, or have a robot carrier take your bags to your room. However, this is not science fiction but tools that hotels are looking to use. Neild from CNN explains that robots are now making an appearance in hotels around the planet - and more are on the way. Marriott hotel in Ghent, Belgium, uses a humanoid robot named Mario, who has been working at the hotel since 2015. Mario is able to welcome his guests in 19 different languages and guards the buffets as well. Even cruise ships are transitioning to similar technologies as Royal Caribbean installed cocktail-mixing robot bartenders on its cruise ships. 

{{< figure src="/images/Robot Image.jpeg" >}}

The Covid-19 pandemic also opened doors to “robot-driven” hotels, as this technology developed sudden importance due to the new demand for minimal human contact interactions with customers. With the accelerated adoption of this technology due to the pandemic, researchers have taken a special interest in the impact robots have on customer satisfaction. This includes researchers Zhong and Verma at Cornell University, who conducted an exploratory study of robots used among hotels in China. They sought to determine whether the guests were pleased with the robots’ assistance. According to their findings, it appears robots are linked to an increase in customer satisfaction. Zhong and Verma administered a survey on 94 guests, half men and half women, across six hotels. They discovered that guests had high expectations for the robots and believed staying in a “robot hotel” would be more cost-effective. Guests felt their stay was more interesting and convenient. But how did the robots do overall in this experiment? They concluded that while not perfect, the robots were also not terrible! Some features were used more frequently by customers than others. For example, turning the lights on and off, turning on the TV, and playing music. Ultimately, nearly all guests rated their overall satisfaction with the robot room as a four or five on a five-point scale. 

With new advancements in text analytics and NLP, we are able to mine for additional information linking robots to customer satisfaction. We plan to utilize and automate these tools to gain actionable insights from Trip Advisor reviews for hotel managers in real-time. We aim to help hotel management improve guest satisfaction by providing them with the most effective model for building a sentiment monitoring system for both human and robot services.

#### Data Processing

This dataset contains 32,829 hotel reviews scraped from the Trip Advisor website.

{{< figure src="/images/Tripadvisor-Logo.png" >}}

As a part of data cleaning, text pre-processing was required before feature engineering and modeling. Text pre-processing is used to bring a standard format to the text. This standardization across a document corpus helps build meaningful features and reduce noise that can be instigated by certain text components, such as irrelevant symbols or special characters. In this project, we applied various text processing skillsets, including pandas, regular expressions, stop word removal, POS tagging, stemming, and lemmatization.

As our next step, we began by splitting our data on the binary “mentions_robots” column. We created a dataset for reviews where robots were mentioned and another where robots were not mentioned. The assumption being that if robots were not mentioned, the service was provided by a human. The split yielded a robot services only dataset and a human services only dataset. Next, we created a binary “high_low” column for rating, where a rating greater than three was classified as a high rating, and a rating of three or less was classified as a low rating. The result was two types of labels in the dataset, high and low. We used this as our label column when modeling. There were 26925 high and 5904 low reviews. The dataset being so unbalanced, we had to perform undersampling to mitigate the skewness before conducting our analyses.


#### Data Analysis

In this project, we employed various techniques to analyze the data. These included natural language processing, machine learning, and deep learning methods. We conducted sentiment analysis for our human and robot datasets. For this, we employed TextBlob. However, we wanted to take it a step further and not just look at sentiment but our ability to predict it. Specifically, the ability to accurately predict rating (high or low) using text alone. Keeping our end goal in mind of finding the most effective model for building a sentiment monitoring system. 

To accomplish this, we utilized machine learning, deep learning, and sentiment scoring models. Our machine learning models included Logistic Regression, Support Vector, Decision Tree, Random Forest, and Multinomial Naive Bayes classifiers. For all the machine learning classifier models, we ran for both count vectorizer and TF-IDF vectorizer. For our deep learning models, we included BERT, Glove, Word2Vec, and FastText. Finally, for our sentiment scoring models, we included SentiWordNet and Vader. 

To evaluate performance, we used accuracy, precision, recall, F1 score, and AUC as our performance metrics. Where TP, TN, FP, and FN represent true positive, true negative, false positive, and false negative, respectively. AUC is another useful metric to validate classification models, as it is threshold and scale-invariant. ROC plots FPR against TPR at different threshold values.

Accuracy = TP+TN/TP+FP+FN+TN

Precision = TP/TP+FP

Recall = TP/TP+FN

F1 Score = 2 * (Recall * Precision) / (Recall + Precision)

TPR (True Positive Rate): TP/(TP+FN)

FPR (False Positive Rate): FP/(FP+TN)

#### Findings

##### General Patterns

{{< figure src="/images/General findings 1.png" >}}

Given the rating score distribution, we can see that hotels are generally doing well. Positive reviews (ratings of 4 or 5) account for 82% of the total reviews. However, we would like to know the factors that contribute to a positive or negative experience. And further, of those factors, what can be leveraged or improved? In answering these questions, we hope to acquire a better understanding of customers and put forward identified business opportunities to stakeholders.

{{< figure src="/images/general findings 2.png" >}}

The word clouds above display a comparison of the robot and human datasets for high and low reviews. In all word clouds, most people seem to base their satisfaction on their stay, room, time, and staff/robot interactions. It appears guests value having a clean room, as the word clean appears in both positive word clouds. Interactions with staff and/or robots also seem to be valued by guests. It is crucial that staff and robots appear friendly and helpful to guests. Time is another significant word in all word clouds. Guests seem to value their time and care about how long interactions take when they need something. Check was another word present in all word clouds, check is referring to check in and check out. These seem to be influential touchpoints with guests, and this makes sense as it is their initial and last impression of the hotel.

##### Sentiment Analysis

In the graphs below, the x-axis shows polarity and the y-axis shows subjectivity. Polarity tells how positive or negative the text is. The subjectivity tells how subjective or opinionated the text is. The green dots that lie on the vertical line are the “neutral” reviews, the red dots on the left are the “negative” reviews, and the blue dots on the right are the “positive” reviews. Bigger dots indicate more subjectivity. We see that positive reviews are more than the negatives in both the human services and robot services datasets. While it is a smaller sample, this is especially true for the robot services dataset. This reveals that not only can our analysis be used for monitoring, but perhaps beyond that used as an argument to convince managers to use robots in their hotels.

{{< figure src="/images/general findings 3.png" >}}
{{< figure src="/images/general findings 4.png" >}}

##### Machine Learning, Deep Learning, Sentiment Scoring Model Performance Comparison

As mentioned previously, after conducting our sentiment analysis in Textblob, we wanted to take the next step to prediction. Specifically, we wanted to test our ability to accurately predict rating (high or low) using text alone. In order to find the best model possible, we tested across machine learning, deep learning, and sentiment scoring models. We created tables which display the calculated scores for accuracy, precision, recall, F1 score, and AUC, and display which  model had the best score in the far right column. Our machine learning models included Logistic Regression, Support Vector, Decision Tree, Random Forest, and Multinomial Naive Bayes classifiers.

Human Services - CV

{{< figure src="/images/human services - cv.png" >}}

Human Services - TF-IDF

{{< figure src="/images/human services - tfidf.png" >}}

Robot Services - CV

{{< figure src="/images/robot services - cv.png" >}}

Robot Services - TF-IDF

{{< figure src="/images/robot services - tfidf.png" >}}


The above tables show the performance of each machine learning model when utilizing either Countvectorizer or TF-IDF vectorizer in predicting high or low ratings in either the human or robot services datasets. For each field, the highest accuracy, precision, recall, F1-score, and AUC metrics are indicated by the “Best Score” column. The highest-performing model for each field fell mostly between the range of 0.7 to 9.0, reflecting relatively accurate classification. For the human services dataset, the Logistic Regression model, with TF-IDF vectorizer, was the best model for predicting high or low rating with an AUC of 0.864 and an accuracy score of 0.864. For the robot services dataset, the Support Vector classifier model, with TF-IDF vectorizer, was also the best model for predicting high or low rating with an AUC of 0.775 and an accuracy score of 0.799. Therefore, of our machine learning models, Logistic Regression was our best performing model for the human dataset and the Support Vector classifier was our best performing model for the robot dataset.

Human Services

{{< figure src="/images/human services - deep.png" >}}

Robot Services

{{< figure src="/images/robot services - deep.png" >}}

For our deep learning models, we included BERT, Glove, Word2Vec, and FastText. The above tables show the performance of each of our word embeddings models in predicting high or low ratings in either the human or robot services datasets. For each field, the highest accuracy, precision, recall, F1-score, and AUC metrics are indicated by the “Best Score” column. The BERT word embeddings model was the highest performing on each metric in the human services dataset and for accuracy and AUC in the robot services dataset. For each field, we found that for the most part the word embeddings outperformed the traditional machine learning Countvectorizer and TF-IDF approach. Overall, the BERT model performed extremely well, never scoring below 0.857 on any metric and scoring over 0.90 in many cases. For the human services dataset, the BERT model was the best model for predicting high or low rating with an AUC of 0.967 and an accuracy score of 0.967. For the robot services dataset, the BERT model was also the best model for predicting high or low rating with an AUC of 0.868 and an accuracy score of 0.868. In each dataset, BERT was the highest-performing option thus far.

Human Services

{{< figure src="/images/human services - senti.png" >}}

Robot Services 

{{< figure src="/images/robot services - senti.png" >}}

Finally, for our sentiment scoring models, we included SentiWordNet and Vader. The above tables show the performance of each of our sentiment scoring models in predicting high or low ratings in either the human or robot services datasets. For each field, the highest accuracy, precision, recall, F1-score, and AUC metrics are indicated by the “Best Score” column. The Vader sentiment scoring model was the highest performing on each metric for both datasets. For the human services dataset, the Vader model was the best model for predicting high or low rating with an accuracy score of 0.866. For the robot services dataset, the Vader model was also the best model for predicting high or low rating with an accuracy score of 0.904. While BERT remains the superior predicting model for the human services dataset, it appears Vader is the best predicting model for the Robot Services dataset. In conclusion, we were able to find a model that performed extremely well for both datasets, with scores all above .90. This reflects our ability to classify high or low ratings accurately.

##### Recommendations

For Managers
* Utilize robots where appropriate - robots clearly have a place in this industry, it is worth finding places to implement

* Train staff to be friendly and helpful - staff and service appeared frequently in the word clouds - the words friendly and helpful were associated in the positive word cloud and are likely  referring to staff

* Focus more on check-in and check-out experience - check appeared in all word clouds, these experiences are clearly important to guests

* Be time efficient - time appeared in both negative word clouds, guests value their time, do not waste it

* Relay feedback to hotel promoters so they can create promotions centered around robots

For Promoters
* Advertise using robots, based on our results robots seem to show positive perception from customers and it is a way hotels can differentiate themselves

* Get feedback from hotel managers on what services are being improved by robots, advertise those services specifically when promoting
