from flask import Flask, render_template, request, redirect
import nlp

app = Flask(__name__)


@app.route('/')
def login():
    return render_template("login.html")


@app.route('/login', methods=['POST'])
def trainee_news():
    if request.form['email']=='producer@training.com':
        nlp.select_yesterday()
        nlp.split_sent()
        nlp.sent2word()
        nlp.vec()
        nlp.tfidf()
        nlp.labeled_data()
        nlp.labeled_vec()
        nlp.transform_vec()
        nlp.predict_nlp_by_zone()
        result=nlp.PRD_msg()
        result=result.values.tolist()
        # result = nlp.mail_msg()
        return render_template("trainee_news.html", result=result)
    else:
        return redirect('/')


@app.route('/fb')
def trainee_news_fb():
    return render_template("trainee_news_fb.html")


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, use_debugger=True)