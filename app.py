from flask import Flask, render_template, request
from sklearn.externals import joblib
from datetime import datetime as dt
import os

app = Flask(__name__, static_url_path='/static/')

# only slash means root folder
@app.route('/')
# rendering an html template that you define
def form():
    return render_template('index.html')


@app.route('/predict_checkouts', methods=['POST', 'GET'])
def predict_price():
    # get the parameters
    shift_1 = float(request.form['shift_1'])
    shift_2 = float(request.form['shift_2'])
    shift_3 = float(request.form['shift_3'])
    shift_4 = float(request.form['shift_4'])
    shift_5 = float(request.form['shift_5'])
    shift_6 = float(request.form['shift_6'])
    shift_7 = float(request.form['shift_7'])
    day_of_week = int(request.form['day_of_week'])
    month = float(request.form['month'])
    year = float(request.form['year'])

    # load the model and predict
    model = joblib.load('model/daily_checkout_model.pkl')
    prediction = model.predict([[shift_1, shift_2, shift_3, shift_4, shift_5, shift_6, shift_7, day_of_week, month, year]])
    predicted_checkouts = prediction.round(0)[0]

    return render_template('results.html',
                           shift_1=int(shift_1),
                           shift_2=int(shift_2),
                           shift_3=int(shift_3),
                           shift_4=int(shift_4),
                           shift_5=int(shift_5),
                           shift_6=int(shift_6),
                           shift_7=int(shift_7),
                           day_of_week=dt.strptime(str(day_of_week + 1), '%d').strftime('%A'),
                           month=int(month),
                           year=int(year),
                           predicted_checkouts="{:,.0f}".format(predicted_checkouts)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
