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
import joblib
from sklearn.ensemble import RandomForestClassifier
import schedule
import time

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

#＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
#６_教師データ
def labeled_data():
    with open('sentence_name_1356.csv',encoding='utf-8') as file:
        dataset=pd.read_table(file, delimiter=',',na_filter=False)

    text_normalized = []
    for text in dataset.sentence_list:
        text_normalized.append(unicodedata.normalize('NFKC', text))
    dataset['text_normalized']=text_normalized

    #形態素解析
    mecab = MeCab.Tagger("-O wakati")
    text_tokenized = []
    for text in dataset['text_normalized']:
        text_tokenized.append(mecab.parse(text))
    dataset['text_tokenized']=text_tokenized

    #数字の統一置き換え
    def normalize_number(text):
        normalize_number_text=[]
        for n in text:
            replaced_text = re.sub(r'\d+', '0', n)
            normalize_number_text.append(replaced_text)
        return normalize_number_text

    text_tokenized_normalized_number = normalize_number(text_tokenized)
    dataset['text_tokenized_normalized_number']=text_tokenized_normalized_number
    labeled=dataset.drop('text_normalized',axis=1)
    labeled=labeled.drop('sentence_list',axis=1)
    labeled=labeled.drop('text_tokenized',axis=1)
    return labeled
labeled=labeled_data()

def labeled_vec():
    #単語行列の作成
    X2 = vectorizer.fit_transform(labeled["text_tokenized_normalized_number"])

    #TFIDF計算
    vectorizert = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizert.fit_transform(labeled["text_tokenized_normalized_number"])
    return X
X=labeled_vec()

#6-2_単語文章行列を学習済みデータ未知データでそれぞれ生成
def transform_vec():
    X_test2 = vectorizer.transform(train["text_tokenized_normalized_number"])
    return X_test2
X_test2=transform_vec()


#＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
#７_学習・予測
#８_予測データ表示
#8-1_ゾーン小さい順に並べ替え
def predict_nlp_by_zone():
    labeled_y = np.array(labeled['zone'])

    #教師データで学習
    rfc = RandomForestClassifier(random_state=1234)
    rfc.fit(X, labeled_y)

    #### モデル読み込み
    rfc_restored = joblib.load("negacheck_fit_1356_2018.pkl.gz")

    y_pred_on_test2=rfc_restored.predict(X_test2)

    test2_ypred= pd.DataFrame(y_pred_on_test2,dtype=int)#arrayからpdに変換
    test2_ypred["No."]=sentense_data['No.']#dayを結合
    test2_ypred["Day"]=sentense_data['day']#dayを結合
    test2_ypred["Name"]=sentense_data['name']#dayを結合
    test2_ypred["sentence"]=sentense_data['sentence_list']#sentenceを結合
    test2_ypred=test2_ypred.rename(columns={0:'nega_predict'})

    day_max_nega=test2_ypred.groupby(['No.','Day']).max()
    day_max_nega=day_max_nega.sort_index()

    predict_data_yesterday=pd.merge(yesterday,day_max_nega,on=['Name','Day'])
    predict_data_yesterday['zone'] = predict_data_yesterday['zone'].astype(int)

    drop_col = ['Active','maemuki','sossenn','control','challenge','jujitu','dekita','sentence_x','sentence_len']
    predict_data_yesterday=predict_data_yesterday.drop(drop_col,axis=1)

    predict_data_yesterday=predict_data_yesterday.sort_values(by=['zone','No.'])#ゾーン小さい順に並べ替え
    return predict_data_yesterday

#8-2_人別に並べ替え
def predict_nlp_by_trainee():
    labeled_y = np.array(labeled['zone'])

    #教師データで学習
    rfc = RandomForestClassifier(random_state=1234)
    rfc.fit(X, labeled_y)

    #### モデル読み込み
    rfc_restored = joblib.load("negacheck_fit_1356_2018.pkl.gz")

    y_pred_on_test2=rfc_restored.predict(X_test2)

    test2_ypred= pd.DataFrame(y_pred_on_test2,dtype=int)#arrayからpdに変換
    test2_ypred["No."]=sentense_data['No.']#dayを結合
    test2_ypred["Day"]=sentense_data['day']#dayを結合
    test2_ypred["Name"]=sentense_data['name']#dayを結合
    test2_ypred["sentence"]=sentense_data['sentence_list']#sentenceを結合
    test2_ypred=test2_ypred.rename(columns={0:'nega_predict'})

    day_max_nega=test2_ypred.groupby(['No.','Day']).max()
    day_max_nega=day_max_nega.sort_index()

    predict_data_yesterday=pd.merge(yesterday,day_max_nega,on=['Name','Day'])
    predict_data_yesterday['zone'] = predict_data_yesterday['zone'].astype(int)

    drop_col = ['Active','maemuki','sossenn','control','challenge','jujitu','dekita','sentence_x','sentence_len']
    predict_data_yesterday=predict_data_yesterday.drop(drop_col,axis=1)

    predict_data_yesterday=predict_data_yesterday.sort_values(by=['No.','Day'])#人別に並べ替え
    return predict_data_yesterday

predict_data_yesterday=predict_nlp_by_zone()


def insert_msg():
    nega2all_msg = f"ネガティブな表現あり。"
    nega2_msg = f"ネガティブな表現あり。"
    nega2_ifnecessary = f"ネガティブな表現あり。"
    zone12nonnega_msg = f"ネガティブな表現はみられませんでしたが、成長が停滞している状態。"
    zone3nonnega_msg = f"ネガティブな表現はみられませんでしたが、成長が停滞している可能性。"
    zone4nonnega_msg = f"順調です。"

    zone1_msg = "アパシーゾーン（無気力）。即時サポートを推奨。"
    zone2_msg = "パニックゾーンの可能性。即時サポートを推奨。"
    zone3_msg = "コンフォートゾーンの可能性。"
    zone4_msg = "順調です（スクワットゾーン）。"
    zone5_msg = "順調です。"
    distortion_msg = "ゾーン判定不可。行動と受講者認識にズレあり。"
    emptyPRD_msg = "Active度の記入をお願いします。"
    emptyTRA_msg = "Dailyアンケート未回答です。"
    bar = "＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝"

    # アパシーゾーンかつコメントネガあり/なし
    predict_data_yesterday.loc[predict_data_yesterday['zone'] == 1, 'zone_msg'] = zone1_msg
    predict_data_yesterday.loc[
        (predict_data_yesterday['zone'] == 1) & (predict_data_yesterday['nega_predict'] > 1), 'nega_msg'] = nega2all_msg
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == 1) & (
                predict_data_yesterday['nega_predict'] < 2), 'nega_msg'] = zone12nonnega_msg
    # パニックゾーンかつコメントネガあり・なし
    predict_data_yesterday.loc[predict_data_yesterday['zone'] == 2, 'zone_msg'] = zone2_msg
    predict_data_yesterday.loc[
        (predict_data_yesterday['zone'] == 2) & (predict_data_yesterday['nega_predict'] > 1), 'nega_msg'] = nega2all_msg
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == 2) & (
                predict_data_yesterday['nega_predict'] < 2), 'nega_msg'] = zone12nonnega_msg
    # コンフォートゾーンかつコメントネガあり・なし
    predict_data_yesterday.loc[predict_data_yesterday['zone'] == 3, 'zone_msg'] = zone3_msg
    predict_data_yesterday.loc[
        (predict_data_yesterday['zone'] == 3) & (predict_data_yesterday['nega_predict'] > 1), 'nega_msg'] = nega2all_msg
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == 3) & (
                predict_data_yesterday['nega_predict'] < 2), 'nega_msg'] = zone3nonnega_msg
    # スクワットゾーンかつコメントネガあり・なし
    predict_data_yesterday.loc[predict_data_yesterday['zone'] == 4, 'zone_msg'] = zone4_msg
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == 4) & (
                predict_data_yesterday['nega_predict'] > 1), 'nega_msg'] = nega2_ifnecessary
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == 4) & (
                predict_data_yesterday['nega_predict'] < 2), 'nega_msg'] = zone4nonnega_msg
    # パフォーマンスゾーン以上かつコメントネガあり・なし
    predict_data_yesterday.loc[predict_data_yesterday['zone'] > 4, 'zone_msg'] = zone5_msg
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] > 4) & (
                predict_data_yesterday['nega_predict'] > 1), 'nega_msg'] = nega2_ifnecessary
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] > 4) & (
                predict_data_yesterday['nega_predict'] < 2), 'nega_msg'] = zone4nonnega_msg
    # ちぐはぐ、かつコメントネガあり
    predict_data_yesterday.loc[predict_data_yesterday['zone'] == 0, 'zone_msg'] = distortion_msg
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == 0) & (
                predict_data_yesterday['nega_predict'] > 1), 'nega_msg'] = nega2_ifnecessary
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == 0) & (
                predict_data_yesterday['nega_predict'] < 2), 'nega_msg'] = zone4nonnega_msg
    # PRD未記入、かつコメントネガあり
    predict_data_yesterday.loc[predict_data_yesterday['zone'] == -1, 'zone_msg'] = emptyPRD_msg
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == -1) & (
                predict_data_yesterday['nega_predict'] > 1), 'nega_msg'] = nega2_ifnecessary
    predict_data_yesterday.loc[(predict_data_yesterday['zone'] == 1) & (
                predict_data_yesterday['nega_predict'] < 2), 'nega_msg'] = zone4nonnega_msg
    # 受講者未回答
    predict_data_yesterday.loc[predict_data_yesterday['zone'] == -2, 'zone_msg'] = emptyTRA_msg

    # 表示用に、列を並べ替え
    PRDmsg = predict_data_yesterday[['No.', 'Name', 'Day', 'zone', 'zone_msg', 'nega_msg', 'comment']]
    PRDmsg = PRDmsg.rename(columns={'Day_x': 'Day', 'zone_msg': '成長ゾーン', 'nega_msg': 'Dailyアンケートコメント判定',
                                    'comment': 'Dailyアンケートコメント引用'})
    pd.options.display.max_colwidth = 500  # 文章が途中で切れないようにするための設定
    PRDmsg = PRDmsg.fillna('No Data')
    return PRDmsg
PRDmsg = insert_msg()


def mail_msg():
    mail_head = f"<p>{company}_{batch}_講師の皆様</p>\
    <p>おはようございます。昨日の受講者の状態をお伝えします。</p><p>Dailyアンケート:" + DR_URL + "</p>"
    mail_end = "<p></p>以上"
    html_sheet = PRDmsg.to_html()
    html = mail_head + html_sheet + mail_end
    return html
html = mail_msg()


def gmail_send():
    from email.utils import make_msgid
    from email import message
    import smtplib

    smtp_host = 'smtp.gmail.com'
    smtp_port = 587
    from_email = ''
    username = ''  # Gmailのアドレス
    password = ''  # Gmailのパスワード

    # メールの内容を作成
    msg = message.EmailMessage()
    msg.set_content(html)  # メールの本文
    yesterday = PRDmsg['Day'][1]
    msg['Subject'] = f"【{yesterday}】{company}_{batch}Trainee news"  # 件名
    msg['From'] = from_email  # メール送信元
    msg['To'] = to_email  # メール送信先
    msg['Cc'] = cc_email

    asparagus_cid = make_msgid()
    msg.add_alternative(html.format(asparagus_cid=asparagus_cid[1:-1]), subtype='html')

    # メールサーバーへアクセス
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(username, password)
    # server.send_message(msg)
    toaddrs = [to_email] + [cc_email]
    server.sendmail(from_email, toaddrs, msg.as_string())
    server.quit()


#送信予約する場合
def notice_PRD():
    schedule.every().day.at("10:00").do(gmail_send)
    while True:
        schedule.run_pending()
        time.sleep(1)
notice_PRD()