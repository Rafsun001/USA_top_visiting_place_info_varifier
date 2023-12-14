# USA Top Visiting Place Information Verifier
Various information about some top popular tourist spot is USA, is collected from various website and this the dataset is prepared. Various data preprocessing techniques are performed on the data and machine learning algorithm is used for classification.

## Tools used:
1. Python 
2. Pandas
3. Numpy
4. Matplotlib
5. Seaborn
6. Scikit-learn
7. BeautifulSoup
8. difflib
9. Nltk

## Data Cleaning
1. Replace certain special characters with their string equivalents.
2, Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases).
3. Removing HTML tags.
4. Remove punctuations.
5. Expanding contractions words.

## Some New Features
1. No of sentence present in a cell.
3. How many numerical values are containing by each sentence.
4. Verb count of the sentence.
4. Adverb count of the sentence.
4. Noun count of the sentence.
4. Adjective count of the sentence.
5. length of sentence of Information column
6. Number of words present in Information column
7. Most occurs verb count
8. Most occurs Adverb
9. Year count of sentence
10. Count of stopwords of a sentence in a cell
 
## Data Preprocessing
CountVectorizer method is used to convert the words into vector.

### How CountVectorizer Works?
The CountVectorizer is a technique used in natural language processing (NLP) for converting a collection of text documents into a matrix of token counts. It's a fundamental tool for text preprocessing before applying machine learning algorithms. 

**Here's how it generally works:**

*1. Breaking Text into Tokens:* The CountVectorizer breaks down the text into individual words or tokens. This process is known as tokenization. It also handles n-grams, which are sequences of words (bi-grams for two words, tri-grams for three words, etc.) as specified by the user.

*2. Building the Vocabulary:* It creates a vocabulary of unique words present in the entire corpus (collection of documents). Each unique word becomes a feature.

*3. Counting Occurrences:* For each document in the corpus, CountVectorizer counts the frequency of each word (or n-gram) in the vocabulary within that document. This results in a matrix where each row corresponds to a document, and each column corresponds to a unique word in the vocabulary. The matrix represents the count of each word's occurrence in each document.

*Example:*

Suppose you have three documents:

Document 1: "I love coding."

Document 2: "Coding is fun."

Document 3: "Python is for coding."

*The CountVectorizer will:*

Create a vocabulary of unique words: ["I", "love", "coding", "is", "fun", "Python", "for"]

*Generate a matrix representation:*

| Document | I | love | Coding | is | fun | Python | for|
| ------ | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Document 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| Document 2 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| Document 3 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |

## Model Training
Logistic Regression algorithm is used to training.

### How Logistic Regression algorithm Works?
In logistic regression, the dependent variable/feature is in binary format.It contains only 0 means false/no/failure etc and 1 means yes/true/success etc. Here data is also arranged like this, some data are for 1(yes) and some data are for 0(no). For example, to identify if the email is spam(1) or not spam(0). Create a curve line from 1 to 0 in a graph. The curve line shape looks like S-shape. Here some data are in binary point 1 and some data are in binary point 0. Now create a curve line over this data. Then the value will be predicted. Now if the data value is less than 0.5 or negative infinity then it will predict the value 0(no) and if the value is greater than equal to 0.5 or positive infinity then it will predict it 1(yes).

**Formula of logistic regression:**

log{Y/(1-Y}=C+B1X1+B2X2+...+BnXn

Here,

Y=Dependent variable which you want to predict

C= y-axis intercept( the point where the line Intercept on the y-axis)

X= X-axis value or independent variable

B=Slope or angle or degree.(This point has the degree of the line which will be drawn.)

![Logistic Regression algorithm](https://github.com/Rafsun001/USA_top_visiting_place_info_varifier/blob/main/Logistic%20Regression%20algorithm.png?raw=true )

#### Working process example:
| Spending time | Click on ads |
| ------ | ----------- |
| 68.88 | No |
| 44.55 | No |
| 30.10 | Yes |
| 29.43 | No |
| 55.45 | Yes |
| 38.45 | Yes |

Here the table has two features one is the spending time of people on a website and the second one is clicking on ads by that person(yes or no). Here click-on adds is depending variable Y and spending time is independent variable X. Here click on adds is in categorical form. So now convert yes as 1 and No as 0. Now if the predicted value is 0 then it means that the person doesn't click on adds and if yes then he clicked. If a scatter plot is drawn then you will see that some values are on the 0 points of the graph and some values are on the 1 point of the graph. Now draw an S-shape curve line on all the data. Now if the data is less than 0.5 or negative infinity then it will predict the value 0(no) and if the value is greater than equal to 0.5 or positive infinity then it will predict it 1(yes).
