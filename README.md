# Natural Language Processing(NLP) 수업 실습
## preprocessing1
+ 토큰화

토큰이라는 단위로 말뭉치를 나누는 작업
한국어에서 어절(띄어쓰기)기반 토큰화는 지양되고 있음->형태소 기반 토큰화

+ 품사 태깅

형태소 단위로 분리 후, 형태소의 품사를 태깅하는 것

+ 어간 추출(Stemming)

문법적으로 다양한 형태의 변종을 단순화 시키는 작업
실제 단어가 아닐 수도 있음

+ 표제어 추출(Lemmatization)

문법적으로 다양한 형태의 변종을 단순화 시키는 작업
단어의 뿌리를 찾는 것(실제 단어 형태)

+ 사용 라이브러리

-영어 단어 토큰화
``` 
import nltk
from nltk.tokenize import word_tokenize as wt 
from nltk.tokenize import WordPunctTokenizer ##모든 구두점을 단위로 분해
from nltk.stem import WordNetLemmatizer as lemma #표제어 추출
from nltk.stem import PorterStemmer as stemming #어간 추출
``` 

-한국어 전처리
```
import konlpy
from konlpy.tag import Kkma
from konlpy.tag import Okt
from konlpy.tag import Hannanum
from konlpy.tag import Komoran
```
## preprocessing2
+ 불용어(stopwords) 처리

유의미한 토큰만을 선별하기 위한 작업

+ 영어 불용어 리스트 불러오기
``` 
from nltk.corpus import stopwords
``` 
## Spam Email Text Data Analysis
#### 1. 라이브러리 호출
### 2. spam.csv(스팸 메일을 가지고 있는 리스트)파일 불러오기
### 3. 결측값(null), 중복값 제거
### 4. 전처리

  4-(1). 토큰화

  4-(2). 품사 태깅

   4-(3). 명사 추출

  4-(4). 표제어 추출(Lemmatization)

  4-(5). 불용어(stop words) 제거

### 5. 문서를 숫자 벡터로 변환(Bag of Words)
### 6. 지도 
## Word2Vec

