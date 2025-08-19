from flask import Flask, request, render_template
from model import SentimentRecommenderModel


app = Flask(__name__)

sentiment_model = SentimentRecommenderModel()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    # Get username from form
    user = request.form['userName']
    user = user.lower()

    # Get recommendations
    items = sentiment_model.getSentimentRecommendations(user)

    if items is not None:
        return render_template(
            "index.html",
            column_names=items.columns.values,
            row_data=list(items.values.tolist()),
            zip=zip,
            userName=user
        )
    else:
        return render_template(
            "index.html",
           message=f"Username '{user}' doesn't exist. No product recommendations at this point of time!",
            userName=user 
        )

if __name__ == '__main__':
    app.run(debug=True)