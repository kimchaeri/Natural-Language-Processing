# Natural Language Processing(NLP) 수업 실습
## preprocessing1
#### 토큰화

토큰이라는 단위로 말뭉치를 나누는 작업
한국어에서 어절(띄어쓰기)기반 토큰화는 지양되고 있음->형태소 기반 토큰화

#### 품사 태깅

형태소 단위로 분리 후, 형태소의 품사를 태깅하는 것

#### 어간 추출(Stemming)

문법적으로 다양한 형태의 변종을 단순화 시키는 작업
실제 단어가 아닐 수도 있음

#### 표제어 추출(Lemmatization)

문법적으로 다양한 형태의 변종을 단순화 시키는 작업
단어의 뿌리를 찾는 것(실제 단어 형태)

#### 사용 라이브러리

-영어 단어 토큰화
``` Python
import nltk
from nltk.tokenize import word_tokenize as wt 
from nltk.tokenize import WordPunctTokenizer ##모든 구두점을 단위로 분해
from nltk.stem import WordNetLemmatizer as lemma #표제어 추출
from nltk.stem import PorterStemmer as stemming #어간 추출
``` 

-한국어 전처리
``` Python
import konlpy
from konlpy.tag import Kkma
from konlpy.tag import Okt
from konlpy.tag import Hannanum
from konlpy.tag import Komoran
```
## preprocessing2
#### 불용어(stopwords) 처리

유의미한 토큰만을 선별하기 위한 작업

#### 영어 불용어 리스트 불러오기
``` Python
from nltk.corpus import stopwords
``` 
## Spam Email Text Data Analysis
#### 1. 라이브러리 호출
``` Python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
``` 
#### 2. spam.csv(스팸 메일을 가지고 있는 리스트)파일 불러오기
``` Python
data_path = r'C:/Users/user/Desktop/딥러닝프레임워크_박성호/spam.csv'
data = pd.read_csv(data_path,encoding='latin1')

``` 
#### 3. 결측값(null), 중복값 제거
``` Python
data.drop_duplicates(subset=['des'],inplace=True, keep='first') #ex.3개가 중복되었을 때 첫 번째것만 남기고 나머지 제거(keep='first') #last or False
``` 
#### 4. 전처리
  4-(1). 토큰화

  4-(2). 품사 태깅

  4-(3). 명사 추출

  4-(4). 표제어 추출(Lemmatization)

  4-(5). 불용어(stop words) 제거
``` Python
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import STOPWORDS

i=1
new_emails=[]
final_tokens=[]

for email in emails:
    #(1) Tokenize
    tokens=tokenizer.tokenize(email)
    
    #(1-1) lower
    tokens=[word.lower() for word in tokens]
    
    #(2) Pos tagging
    pos_results=pos_tag(tokens)
    
    #(3) Noun Extract
    #lemma_noun=list()
    #stem_noun=list()
    
    temp_tokens=[]
    for word, tag in pos_results:
        
        #(4) lemma tizing
        lemma=Lemmatizer.lemmatize(word)
        
        #(5) stop word
        if lemma not in stop_words:
            final_tokens.append(lemma)
            temp_tokens.append(lemma)
    new_emails.append(temp_tokens)
    if i%100==0:
        print(i)
        
    i+=1
```
#### 5. 문서를 숫자 벡터로 변환(Bag of Words)
``` Python
from sklearn.feature_extraction.text import CountVectorizer
TF=CountVectorizer() #각 텍스트에서 단어 출현 횟수를 카운팅한 벡터
TF_matrix=TF.fit_transform(new_str_emails3) #코퍼스로부터 각 단어의 빈도 수를 기록
```
#### 6. 지도 학습
``` Python
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(TF_matrix.drop(['Y'],axis=1),TF_matrix['Y'],test_size=0.30, stratify=Y, random_state=2313, shuffle=True)
#입력값으로는 원본 데이터의 x,y
#test_size: 테스트 셋 구성의 비율
#shuffle: default=True 입니다. split을 해주기 이전에 섞을건지 여부
#stratify: default=None 입니다. classification을 다룰 때 매우 중요한 옵션값, stratify 값을 target으로 지정해주면 각각의 class 비율(ratio)을 train / validation에 유지해 줍니다. (한 쪽에 쏠려서 분배되는 것을 방지합니다) 만약 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,balanced_accuracy_score
from sklearn.metrics import classification_report,recall_score

RF_test_pred=RF.predict(test_x)
GB_test_pred=GB.predict(test_x)
LR_test_pred=LR.predict(test_x)

print('RF_confusion',confusion_matrix(test_y,RF_test_pred))
print('RF_accuracy',accuracy_score(test_y,RF_test_pred))
print('RF_recall',recall_score(test_y,RF_test_pred))
print('RF_balanced',balanced_accuracy_score(test_y,RF_test_pred))
print('##############################################################')

print('GB_confusion',confusion_matrix(test_y,GB_test_pred))
print('GB_accuracy',accuracy_score(test_y,GB_test_pred))
print('GB_recall',recall_score(test_y,GB_test_pred))
print('GB_balanced',balanced_accuracy_score(test_y,GB_test_pred))
print('##############################################################')

print('LR_confusion',confusion_matrix(test_y,LR_test_pred))
print('LR_accuracy',accuracy_score(test_y,LR_test_pred))
print('LR_recall',recall_score(test_y,LR_test_pred))
print('LR_balanced',balanced_accuracy_score(test_y,LR_test_pred))
print('##############################################################')
```
## Word2Vec

