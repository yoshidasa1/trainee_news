#NLP（自然言語処理）の処理。これをtrainee_news.pyでimportして実行できないかと考えています

#pycharmでのパッケージのインストールが、一部、うまく検索で見つけられなかったものがありました。pickleやreなど
#機械学習前の前処理の途中で、書きかけですが、コードの書き方や関数の動かし方など、アドバイスいただければ嬉しいです

from __future__ import print_function
import pickle
import os.path
from googleapiclient import discovery
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import numpy as np
import pandas as pd
import re
import unicodedata
import MeCab
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

###########################################
#案件情報　毎回作成
company='Z社様'
batch='2nd_2019'

#案件DailyアンケートURL
DR_URL = "https://docs.google.com/spreadsheets/d/1bNbYGN1ZlA_q0g7ES4uKsMz62cXD0WnUhXBuQj--op8/edit#gid=1737347684"
DR_sheetkey = "1bNbYGN1ZlA_q0g7ES4uKsMz62cXD0WnUhXBuQj--op8"
DR_sheetrange = '回答一覧表!A3:N182'

#案件講師メールアドレス
test='hommyo@alue.co.jp'
prd1_mail='global_producer1@alue-training.com'

to_email = prd1_mail # 送りたい先のアドレス
cc_email = test # 送りたい先のアドレス

######################################################
SCOPES = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive','https://www.googleapis.com/auth/drive.file']

# スプレッドシート読み込み
def main():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server()
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=DR_sheetkey,
                                range=DR_sheetrange).execute()
    values = result.get('values', [])
    data = pd.DataFrame(values,
                        columns=['No.', 'sheet', 'Name', 'Day', 'sheetkey', 'Active', 'maemuki', 'sossenn', 'control',
                                 'challenge', 'jujitu', 'dekita', 'comment', 'zone'])
    return data

if __name__ == '__main__':
    main()

#2-1_昨日のDayだけ取得
def select_yesterday():
    data=main()
    drop_cols = ['day','sheet']
    data=data.drop(drop_cols,axis=1)
    data.comment[data.comment == ''] = 'NoData'
    kako=data[data['comment']!='NoData'].max()
    recent=kako['Day']
    yesterday=data[data['Day']==recent]
    yesterday=yesterday.reset_index(drop=True)
    return yesterday

yesterday=select_yesterday()

#3_　文章分割
def split_sent():
    #3-1_改行コードを消す
    box = []
    for k in yesterday["comment"]:
        split_sentence = re.sub(r"\s+","",k)
        box.append(split_sentence)
    comment2 =  pd.DataFrame(box,columns=['comment'])
    comment=comment2['comment']

    #3-2_1文ずつに区切る処理
    sentence=[]
    for i in range(len(comment)):
        s=comment[i].split("。")
        sentence.append([x for x in s if x])
    yesterday['sentence']=sentence
    yesterday['sentence_len'] = yesterday['sentence'].apply(len)

    #3-3_データ結合
    trainee_number_list = []#No.リストの作成
    for i in range(len(yesterday)):
        tmp = [yesterday['No.'][i]] * yesterday['sentence_len'][i]
        trainee_number_list.extend(tmp)
    day_list = []#DAYリストの作成
    for i in range(len(yesterday)):
        tmp = [yesterday['Day'][i]] * yesterday['sentence_len'][i]
        day_list.extend(tmp)
    name_list = []#Nameリストの作成
    for i in range(len(yesterday)):
        tmp = [yesterday['Name'][i]] * yesterday['sentence_len'][i]
        name_list.extend(tmp)
    sentence_list = []#文リストの作成
    for i in range(len(yesterday)):
        sentence_list.extend(yesterday['sentence'][i])

    sentence_data = pd.DataFrame({
        'No.': trainee_number_list,
        'day': day_list,
        'name': name_list,
        'sentence_list': sentence_list
        }) # 上で作成したリストを結合してデータフレームを作成する
    # #改行が余分にあるので、削除。文リスト=""＝Trueとして抽出した条件をbool_listとし、~で反対のFalseを取得したもの
    bool_list = sentence_data["sentence_list"]==""
    sentense_data=sentence_data[~bool_list]
    return sentense_data
sentense_data=split_sent()

#３_単語分割
def sent2word():
    #3-4_単語の正規化
    text_normalized = []
    for text in sentense_data.sentence_list:
        text_normalized.append(unicodedata.normalize('NFKC', text))
    sentense_data['text_normalized']=text_normalized
    #3-5_形態素解析
    mecab = MeCab.Tagger("-O wakati")
    text_tokenized = []
    for text in sentense_data['text_normalized']:
        text_tokenized.append(mecab.parse(text))
    sentense_data['text_tokenized']=text_tokenized
    #3-6_数字の統一置き換え
    def normalize_number(text):
        normalize_number_text=[]
        for n in text:
            replaced_text = re.sub(r'\d+', '0', n)
            normalize_number_text.append(replaced_text)
        return normalize_number_text
    text_tokenized_normalized_number = normalize_number(text_tokenized)

    sentense_data["text_tokenized_normalized_number"]=text_tokenized_normalized_number
    train=sentense_data.drop('sentence_list',axis=1)
    train=train.drop('text_normalized',axis=1)
    train=train.drop('text_tokenized',axis=1)
    return train
train=sent2word()

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# ４_ベクトル
# 4-1_単語行列の作成
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
def vec():
    X = vectorizer.fit_transform(train["text_tokenized_normalized_number"])
    return X
X=vec()

# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# ５_TFIDF計算
def tfidf():
    vectorizert = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizert.fit_transform(train["text_tokenized_normalized_number"])
    return X
X=tfidf()