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
#### 1. 라이브러리 호출
``` Python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import time
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
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
  4-(1). 대문자->소문자

  4-(2). 특수 기호, 구두점 제거

  4-(3). 토큰화

``` Python
#(1)
#대문자->소문자
#특수기호 구두점 등 제거

normalized_text=[] #전처리된 텍스트

for string in data['des']:
    tokens=re.sub(r"[^a-z0-9]+"," ",string.lower()) #대문자를 소문자로 바꾼후, 스팸이메일에서 소문자영어나 숫자가 아닌경우 빈 공간으로
    normalized_text.append(tokens)
    
#(2)
#단어 토큰화

#normalized_text에서 각각의 이메일을 sentence로 받아서 work_tokenize시키겠다는 것
result=[word_tokenize(sentence) for sentence in normalized_text]
```
#### 5. Word2Vec 알고리즘 학습 및 실행 by gensim
``` Python
#size=embedding 차원
#window=문맥 크기
#min_count=단어 최소 빈도 수 제한 (빈도가 적은 단어들은)
#workers=학습을 위한 프로세스 수
#sg=0은 CBOW, 1은 Skip-gram
#size:100개 짜리의 벡터 크기로 요약
#window:양 옆에 다섯개씩을 보겠다는 것

model2=Word2Vec(sg=0,size=100,window=5,min_count=1,workers=4)
model2.build_vocab(sentences=result)
```
#### 6. Pretrained된 모델 사용, Transfer Learning
``` Python
import urllib.request
urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",filename="GoogleNews-vectors-negative300.bin.gz")

#Fine Tuning할 새로운 Word2Vec 모델 생성
#PreTrainedKeyvector와 vector_size'가 같은 word2vec model을 생성
TransferedModel2=Word2Vec(size=PreTrainedKeyvector.vector_size,min_count=1)

#단어 생성(build_vocab) by PreTrainedKeyvector word Vocabulary
#TransferedModel.build_vocab input:
#[[]] #list of list
TransferedModel2.build_vocab([PreTrainedKeyvector.vocab.keys()])

#주어진 데이터로 새로운 모델의 단어 추가
#update parameter를 True로 설정
TransferedModel2.build_vocab(result,update=True)

#새로운 데이터들의 단어(토큰) 기반 fine tuning->Training
TransferedModel2.train(result,total_examples=len(result),epochs=1)
```

## Text Classification by Word2Vec and CNN
#### 1. 라이브러리 호출
``` Python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
import time

#Word2Vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#Neural Network
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
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
  4-(1). 대문자->소문자

  4-(2). 특수 기호, 구두점 제거

  4-(3). 토큰화

``` Python
#(1)
#대문자->소문자
#특수기호 구두점 등 제거

normalized_text=[] #전처리된 텍스트

for string in data['des']:
    tokens=re.sub(r"[^a-z0-9]+"," ",string.lower()) #대문자를 소문자로 바꾼후, 스팸이메일에서 소문자영어나 숫자가 아닌경우 빈 공간으로
    normalized_text.append(tokens)
    
#(2)
#단어 토큰화

#normalized_text에서 각각의 이메일을 sentence로 받아서 work_tokenize시키겠다는 것
result=[word_tokenize(sentence) for sentence in normalized_text]
```
#### 5. Pretrained된 모델 사용, Transfer Learning
``` Python
#LOAD pre-trained key vector
#model을 load한 것이 아니고 Embedding vector만 load
#limit=단어수 조정(빅데이터의 경우)
PreTrainedKeyvector=KeyedVectors.load_word2vec_format(
    googleNews_filepath, binary=True, limit=5000 #5000개의 단어만 불러올 것
)

#Fine tuning 할 새로운 Word2Vec 모델 생성
#PreTrainedKeyvector와 'vector_size'가 같은 word2vec model을 생성
#workers=-1 가용할 수 있는 모든 코어 수
TransferedModel=Word2Vec(size=PreTrainedKeyvector.vector_size,min_count=1, workers=-1)

#단어 생성(build_vocab) by PreTrainedKeyvector word Vocabulary
#TransferedModel.build_vocab input:
#[[]] #list of list
TransferedModel.build_vocab([PreTrainedKeyvector.vocab.keys()])

#주어진 데이터로 새로운 모델의 단어 추가
#update parameter를 True로 설정
TransferedModel.build_vocab(result,update=True)

#Pretrained 모델의 학습 파라미터를 기반으로 새로운 모델의 학습 파라미터 초기화
#학습파라미터를 'googleNews_filepath'에 있는 값으로 모두 업데이트해줌
#lockf=0.0 : 보통은 학습 파라미터를 update하지 못하도록 lock이 걸려있음
#lockf=1 : 학습 파라미터를 update하도록 lock 해제
TransferedModel.intersect_word2vec_format(
    googleNews_filepath, binary=True, lockf=1.0
)

#모델 업데이트 - Fine Tuning
#새로운 데이터 기반의 학습
#epochs이 많으면 많을수록 학습을 많이 하는 것
TransferedModel.train(result, total_examples=len(result), epochs=100)
```
#### 6. Text 정수화
``` Python
word_to_id = dict()
id_to_word = dict()
all_words=TransferedModel.wv.vocab.keys()

for word in all_words:
    if word not in word_to_id:
        new_id=len(word_to_id)+1
        word_to_id[word]=new_id
        id_to_word[new_id]=word
        
#각 단어를 이미 정해진 인덱스로 변환
encoded_result=[]
for sent in result:
    temp_id=[]
    for word in sent:
        temp_id.append(word_to_id[word])
    encoded_result.append(temp_id)
```
#### 7. 고정된 길이 padding
``` Python
#post: 뒷 부분에 0을 채워넣겠다는 것
#maxlen: 최대 길이
padded_encoded_result=pad_sequences(encoded_result,padding='post')
```
#### 8. Train, Test 분리
``` Python
#indices: 분할시킬 데이터를 입력
#test_size: 테스트 셋 구성의 비율
#shuffle: 뒤죽 박죽 섞는 것
#random_state: 세트를 섞을 때 해당 int 값을 보고 섞음
#stratify: 이 값을 target으로 지정해주면 각각의 class 비율(ratio)을 train/validation에 유지해 줍니다.(한 쪽에 쏠려서 분배되는 것을 방지)
indices_train, indices_test = train_test_split(indices, test_size=0.2, shuffle=True, random_state=0, stratify=Y)

train_X=padded_encoded_result[indices_train]
train_Y=Y.iloc[indices_train]

test_X=padded_encoded_result[indices_test]
test_Y=Y.iloc[indices_test]
```
#### 9. CNN 구축
``` Python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D#Conv2D
from tensorflow.keras.layers import GlobalMaxPooling1D

#vocab_size: row의 길이
#TransferedModel.wv.vector_size: column의 길이
#원래 weights를 알아서 학습시키는 것이지만 우리는 이미 있는것을 가져다 쓸 것
#trainable=False: embedding_matrix를 train시킬 것이냐 라고 했을 때 안한다는 것
embedding_layer=Embedding(vocab_size, TransferedModel.wv.vector_size, weights=[embedding_matrix], input_length=max_len, trainable=False)

#Architecture 구성
#Conv1D로 하면 input에 맞게끔 너비가 결정, 높이만 결정하면 됨

CNN=Sequential() #순서대로 구성하겠다
CNN.add(embedding_layer)
CNN.add(Conv1D(filters=50, kernel_size=1, activation='relu')) #필터의 개수 50개, kernel_size는 높이(1칸), kernel_size=1이면 190X1짜리가 50개 생김
CNN.add(GlobalMaxPooling1D()) #가장 큰 값을 뽑아서 하나로 요약(50개 만들어짐)
CNN.add(Flatten())
CNN.add(Dense(1, activation='sigmoid')) #fully-connected->Dense, output이 한개
print(CNN.summary())

#학습 요소 구성
#loss: 어떤 손실 함수를 쓸 것인지
#optimizer: adam방식의 경사하강법 사용
#metrics: 이 모델이 잘 구축되었는지 아닌지 볼 지표
CNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#x: Input
#y: Output
#epochs: 반복 횟수
#w를 한 번 업데이트할 때 배치사이즈 만큼의 데이터를 사용
CNN.fit(x=train_X, y=np.array(train_Y), epochs=1, verbose=1, batch_size=32, validation_data=(test_X, np.array(test_Y)))
```
## RNN
#### 1. 라이브러리 호출
``` Python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
```
#### 2. RNN 구축
``` Python
embedding_layer=Embedding(vocab_size, 4, trainable=True, input_length=max_len) #4개짜리 크기의 벡터

embedding_dim=4
hidden_units=10

RNN=Sequential()
RNN.add(embedding_layer)
RNN.add(SimpleRNN(units=hidden_units, activation='relu')) #units:히든 레이어의 히든 노드 개수
RNN.add(Dense(vocab_size, activation='softmax')) #9개의 단어에 대한 확률값을 뱉어줌
RNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #멀티 클래스 문제기이 때문에 categorical_crossentropy 사용
RNN.fit(X,y_cate, epochs=200, verbose=1)
```
