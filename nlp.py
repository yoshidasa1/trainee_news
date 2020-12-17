import pickle
import re
import time
import unicodedata

import MeCab
import joblib
import numpy as np
import pandas as pd
# import schedule
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from email.utils import make_msgid
from email import message
import smtplib
import pj_info

#DBはgoogleドライブで代用していたため、google 認証などの手順があるが、省略
# SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive',
#           'https://www.googleapis.com/auth/drive.file']
# # スプレッドシート読み込み
# def main():
#     creds = None
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server()
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
#
#     service = build('sheets', 'v4', credentials=creds)
#
#     sheet = service.spreadsheets()
#     result = sheet.values().get(spreadsheetId=DR_SHEETKEY,
#                                 range=DR_SHEETRANGE).execute()
#     values = result.get('values', [])
#     data = pd.DataFrame(values,
#                         columns=['No.', 'sheet', 'Name', 'Day', 'sheetkey', 'Active', 'maemuki', 'sossenn', 'control',
#                                  'challenge', 'jujitu', 'dekita', 'comment', 'zone'])
#     return data
#
#
# 2-1_昨日のDayだけ取得
def select_yesterday():
    with open('csv/rawdata.csv', encoding='utf-8') as file:
        data = pd.read_table(file, delimiter=',', na_filter=False)
    drop_cols = ['sheet', 'sheetkey']
    data = data.drop(drop_cols, axis=1)
    # data.comment[data.comment == ''] = 'NoData'
    data.comment[data.loc[:,'comment'] == ''] = 'NoData'
    kako = data[data['comment'] != 'NoData'].max()
    recent = kako['Day']
    yesterday = data[data['Day'] == recent]
    yesterday = yesterday.reset_index(drop=True)
    # 3-1_改行コードを消す
    box = []
    for k in yesterday["comment"]:
        split_sentence = re.sub(r"\s+", "", k)
        box.append(split_sentence)
    comment2 = pd.DataFrame(box, columns=['comment'])
    comment = comment2['comment']
    # 3-2_1文ずつに区切る処理
    sentence = []
    for i in range(len(comment)):
        s = comment[i].split("。")
        sentence.append([x for x in s if x])
    yesterday['sentence'] = sentence
    yesterday['sentence_len'] = yesterday['sentence'].apply(len)
    return yesterday

#
# 3_　文章分割
def split_sent():
    yesterday = select_yesterday()
    # 3-3_データ結合
    trainee_number_list = [[yesterday['No.'][i]] * yesterday['sentence_len'][i] for i in range(len(yesterday))]
    trainee_number_list = [z for i in trainee_number_list for z in i]
    day_list = [[yesterday['Day'][i]] * yesterday['sentence_len'][i] for i in range(len(yesterday))]
    day_list = [z for i in day_list for z in i]
    name_list = [[yesterday['Name'][i]] * yesterday['sentence_len'][i] for i in range(len(yesterday))]
    name_list = [z for i in name_list for z in i]
    sentence_list = [yesterday['sentence'][i] for i in range(len(yesterday))]
    sentence_list = [z for i in sentence_list for z in i]

    sentence_data = pd.DataFrame({
        'No.': trainee_number_list,
        'day': day_list,
        'name': name_list,
        'sentence_list': sentence_list
    })  # 上で作成したリストを結合してデータフレームを作成する
    # #改行が余分にあるので、削除。文リスト=""＝Trueとして抽出した条件をbool_listとし、~で反対のFalseを取得したもの
    bool_list = sentence_data["sentence_list"] == ""
    sentense_data = sentence_data[~bool_list]
    return sentense_data


# ３_単語分割
def sent2word():
    # 3-4_単語の正規化
    text_normalized = []
    sentense_data = split_sent()
    for text in sentense_data.sentence_list:
        text_normalized.append(unicodedata.normalize('NFKC', text))
    sentense_data['text_normalized'] = text_normalized
    # 3-5_形態素解析
    mecab = MeCab.Tagger("-O wakati")
    text_tokenized = []
    for text in sentense_data['text_normalized']:
        text_tokenized.append(mecab.parse(text))
    sentense_data['text_tokenized'] = text_tokenized

    # 3-6_数字の統一置き換え
    def normalize_number(text):
        normalize_number_text = []
        for n in text:
            replaced_text = re.sub(r'\d+', '0', n)
            normalize_number_text.append(replaced_text)
        return normalize_number_text

    text_tokenized_normalized_number = normalize_number(text_tokenized)

    sentense_data["text_tokenized_normalized_number"] = text_tokenized_normalized_number
    drop_cols = ['sentence_list', 'text_normalized', 'text_tokenized']
    train = sentense_data.drop(drop_cols, axis=1)
    return train


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# ４_ベクトル
# 4-1_単語行列の作成
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
def vec():
    train = sent2word()
    X = vectorizer.fit_transform(train["text_tokenized_normalized_number"])
    return X
# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# ５_TFIDF計算
vectorizert = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
def tfidf():
    train = sent2word()
    X = vectorizert.fit_transform(train["text_tokenized_normalized_number"])
    return X


X = tfidf()


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# ６_教師データ
def labeled_data():
    with open('csv/sentence_name_1356.csv', encoding='utf-8') as file:
        dataset = pd.read_table(file, delimiter=',', na_filter=False)
    text_normalized = []
    for text in dataset.sentence_list:
        text_normalized.append(unicodedata.normalize('NFKC', text))
    dataset['text_normalized'] = text_normalized

    # 形態素解析
    mecab = MeCab.Tagger("-O wakati")
    text_tokenized = []
    for text in dataset['text_normalized']:
        text_tokenized.append(mecab.parse(text))
    dataset['text_tokenized'] = text_tokenized

    # 数字の統一置き換え
    def normalize_number(text):
        normalize_number_text = []
        for n in text:
            replaced_text = re.sub(r'\d+', '0', n)
            normalize_number_text.append(replaced_text)
        return normalize_number_text

    text_tokenized_normalized_number = normalize_number(text_tokenized)
    dataset['text_tokenized_normalized_number'] = text_tokenized_normalized_number
    drop_cols = ['sentence_list', 'text_normalized', 'text_tokenized']
    labeled = dataset.drop(drop_cols, axis=1)
    return labeled


def labeled_vec():
    labeled = labeled_data()
    # 単語行列の作成
    X2 = vectorizer.fit_transform(labeled["text_tokenized_normalized_number"])
    # TFIDF計算
    X = vectorizert.fit_transform(labeled["text_tokenized_normalized_number"])
    return X


# 6-2_単語文章行列を学習済みデータ未知データでそれぞれ生成
def transform_vec():
    train = sent2word()
    X_test2 = vectorizer.transform(train["text_tokenized_normalized_number"])
    return X_test2


# ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
# ７_学習・予測
# ８_予測データ表示
# 8-1_ゾーン小さい順に並べ替え
def predict_nlp_by_zone():
    labeled = labeled_data()
    labeled_y = np.array(labeled['zone'])
    yesterday = select_yesterday()
    sentense_data = split_sent()
    X = labeled_vec()
    X_test2 = transform_vec()
    # 教師データで学習
    rfc = RandomForestClassifier(random_state=1234)
    rfc.fit(X, labeled_y)
    #### モデル読み込み
    # rfc_restored = joblib.load("model/negacheck_fit_1356_2018.pkl.gz")
    y_pred_on_test2 = rfc.predict(X_test2)
    test2_ypred = pd.DataFrame(y_pred_on_test2, dtype=int)  # arrayからpdに変換
    test2_ypred["No."] = sentense_data['No.']  # No.を結合
    test2_ypred["Day"] = sentense_data['day']  # dayを結合
    test2_ypred["Name"] = sentense_data['name']  # nameを結合
    test2_ypred["sentence"] = sentense_data['sentence_list']  # sentenceを結合
    test2_ypred = test2_ypred.rename(columns={0: 'nega_predict'})

    day_max_nega = test2_ypred.groupby(['No.', 'Day']).max()
    day_max_nega = day_max_nega.sort_index()

    predict_data_yesterday = pd.merge(yesterday, day_max_nega, on=['Name', 'Day'])
    predict_data_yesterday['zone'] = predict_data_yesterday['zone'].astype(int)

    drop_col = ['Active', 'maemuki', 'sossenn', 'control', 'challenge', 'jujitu', 'dekita', 'sentence_x', 'sentence_len']
    predict_data_yesterday = predict_data_yesterday.drop(drop_col, axis=1)

    predict_data_yesterday = predict_data_yesterday.sort_values(by=['zone', 'No.'])  # ゾーン小さい順に並べ替え
    return predict_data_yesterday


# 8-2_人別に並べ替え
def predict_nlp_by_trainee():
    labeled = labeled_data()
    labeled_y = np.array(labeled['zone'])
    yesterday = select_yesterday()
    sentense_data = split_sent()
    X = labeled_vec()
    X_test2 = transform_vec()
    labeled_y = np.array(labeled['zone'])
    yesterday = select_yesterday()
    sentense_data = split_sent()
    # 教師データで学習
    rfc = RandomForestClassifier(random_state=1234)
    rfc.fit(X, labeled_y)
    #### モデル読み込み
    # rfc_restored = joblib.load("model/negacheck_fit_1356_2018.pkl.gz")
    y_pred_on_test2 = rfc.predict(X_test2)

    test2_ypred = pd.DataFrame(y_pred_on_test2, dtype=int)  # arrayからpdに変換
    test2_ypred["No."] = sentense_data['No.']  # dayを結合
    test2_ypred["Day"] = sentense_data['day']  # dayを結合
    test2_ypred["Name"] = sentense_data['name']  # dayを結合
    test2_ypred["sentence"] = sentense_data['sentence_list']  # sentenceを結合
    test2_ypred = test2_ypred.rename(columns={0: 'nega_predict'})

    day_max_nega = test2_ypred.groupby(['No.', 'Day']).max()
    day_max_nega = day_max_nega.sort_index()

    predict_data_yesterday = pd.merge(yesterday, day_max_nega, on=['Name', 'Day'])
    predict_data_yesterday['zone'] = predict_data_yesterday['zone'].astype(int)

    drop_col = ['Active', 'maemuki', 'sossenn', 'control', 'challenge', 'jujitu', 'dekita', 'sentence_x',
                'sentence_len']
    predict_data_yesterday = predict_data_yesterday.drop(drop_col, axis=1)

    predict_data_yesterday = predict_data_yesterday.sort_values(by=['No.', 'Day'])  # 人別に並べ替え
    return predict_data_yesterday

def insert_msg(predict_data_yesterday):
    nega2all_msg = f"ネガティブな表現あり。コメントを参照ください"
    nega2_ifnecessary = f"ネガティブな表現あり。必要に応じてサポート"
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
    for j in range(len(predict_data_yesterday)):
        predict_data_yesterday['zone_msg']=""
        predict_data_yesterday['nega_msg']=""
    for i in range(len(predict_data_yesterday)):
        if predict_data_yesterday['zone'][i] == 1:
            predict_data_yesterday['zone_msg'][i] = zone1_msg
            if predict_data_yesterday['nega_predict'][i] > 1:
                predict_data_yesterday['nega_msg'][i] = nega2all_msg
            elif predict_data_yesterday['nega_predict'][i] < 2:
                predict_data_yesterday['nega_msg'][i] = zone12nonnega_msg
        elif predict_data_yesterday['zone'][i] == 2:
            predict_data_yesterday['zone_msg'][i] = zone2_msg
            if predict_data_yesterday['nega_predict'][i] > 1:
                predict_data_yesterday['nega_msg'][i] = nega2all_msg
            elif predict_data_yesterday['nega_predict'][i] < 2:
                predict_data_yesterday['nega_msg'][i] = zone12nonnega_msg
        elif predict_data_yesterday['zone'][i] == 3:
            predict_data_yesterday['zone_msg'][i] = zone3_msg
            if predict_data_yesterday['nega_predict'][i] > 1:
                predict_data_yesterday['nega_msg'][i] = nega2all_msg
            elif predict_data_yesterday['nega_predict'][i] < 2:
                predict_data_yesterday['nega_msg'][i] = zone3nonnega_msg
        elif predict_data_yesterday['zone'][i] == 4:
            predict_data_yesterday['zone_msg'][i] = zone4_msg
            if predict_data_yesterday['nega_predict'][i] > 1:
                predict_data_yesterday['nega_msg'][i] = nega2_ifnecessary
            elif predict_data_yesterday['nega_predict'][i] < 2:
                predict_data_yesterday['nega_msg'][i] = zone4nonnega_msg
        elif predict_data_yesterday['zone'][i] > 4:
            predict_data_yesterday['zone_msg'][i] = zone5_msg
            if predict_data_yesterday['nega_predict'][i] > 1:
                predict_data_yesterday['nega_msg'][i] = nega2_ifnecessary
            elif predict_data_yesterday['nega_predict'][i] < 2:
                predict_data_yesterday['nega_msg'][i] = zone4nonnega_msg
        elif predict_data_yesterday['zone'][i] == 0:
            predict_data_yesterday['zone_msg'][i] = distortion_msg
            if predict_data_yesterday['nega_predict'][i] > 1:
                predict_data_yesterday['nega_msg'][i] = nega2_ifnecessary
            elif predict_data_yesterday['nega_predict'][i] < 2:
                predict_data_yesterday['nega_msg'][i] = zone4nonnega_msg
        elif predict_data_yesterday['zone'][i] == -1:
            predict_data_yesterday['zone_msg'][i] = emptyPRD_msg
            if predict_data_yesterday['nega_predict'][i] > 1:
                predict_data_yesterday['nega_msg'][i] = nega2_ifnecessary
            elif predict_data_yesterday['nega_predict'][i] < 2:
                predict_data_yesterday['nega_msg'][i] = zone4nonnega_msg
        elif predict_data_yesterday['zone'][i] == -2:
            predict_data_yesterday['zone_msg'][i] = emptyTRA_msg
    return predict_data_yesterday


def PRD_msg():
    # ゾーン判定、ネガ判定に沿って、コメント列を挿入
    predict_data_yesterday = predict_nlp_by_zone()
    predict_data_yesterday = insert_msg(predict_data_yesterday)
    # 表示用に、列を並べ替え
    PRDmsg = predict_data_yesterday[['No.', 'Name', 'Day', 'zone', 'zone_msg', 'nega_msg', 'comment']]
    PRDmsg = PRDmsg.rename(columns={'Day_x': 'Day', 'zone_msg': '成長ゾーン', 'nega_msg': 'Dailyアンケートコメント判定',
                                    'comment': 'Dailyアンケートコメント引用'})
    pd.options.display.max_colwidth = 500  # 文章が途中で切れないようにするための設定
    PRDmsg = PRDmsg.fillna('No Data')
    return PRDmsg


def mail_msg():
    PRDmsg = PRD_msg()
    mail_head = f"<p>{pj_info.COMPANY}_{pj_info.BATCH}_講師の皆様</p>\
    <p>おはようございます。昨日の受講者の状態をお伝えします。</p><p>Dailyアンケート:" + pj_info.DR_URL + "</p>"
    mail_end = "<p></p>以上"
    html_sheet = PRDmsg.to_html()
    html = mail_head + html_sheet + mail_end
    return html


def gmail_send():
    smtp_host = 'smtp.gmail.com'
    smtp_port = 587
    from_email = ''
    username = ''  # mailのアドレス
    password = ''  # mailのパスワード

    # メールの内容を作成
    msg = message.EmailMessage()
    html = mail_msg()
    PRDmsg = PRD_msg()
    msg.set_content(html)  # メールの本文
    yesterday = PRDmsg['Day'][1]
    msg['Subject'] = f"【{yesterday}】{pj_info.COMPANY}_{pj_info.BATCH}Trainee news"  # 件名
    msg['From'] = from_email  # メール送信元
    msg['To'] = pj_info.to_email  # メール送信先
    msg['Cc'] = pj_info.cc_email

    asparagus_cid = make_msgid()
    msg.add_alternative(html.format(asparagus_cid=asparagus_cid[1:-1]), subtype='html')

    # メールサーバーへアクセス
    server = smtplib.SMTP(smtp_host, smtp_port)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(username, password)
    # server.send_message(msg)
    toaddrs = [pj_info.to_email] + [pj_info.cc_email]
    server.sendmail(from_email, toaddrs, msg.as_string())
    server.quit()
gmail_send()

# 送信予約する場合
def notice_PRD():
    schedule.every().day.at("10:00").do(gmail_send)
    while True:
        schedule.run_pending()
        time.sleep(1)


notice_PRD()


if __name__ == '__main__':
    select_yesterday()
