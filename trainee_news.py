from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def trainee_news():
    return render_template("trainee_news.html")


@app.route('/fb')
def trainee_news_fb():
    return render_template("trainee_news_fb.html")


if __name__ == '__main__':
    app.run()