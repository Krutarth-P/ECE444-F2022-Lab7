from flask import Flask, request, json, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = Flask(__name__)

#class for model
class Model(object):

    # model loading
    loaded_model = None

    @classmethod
    def init_model(self):
        with open('./models/basic_classifier.pkl', 'rb') as fid:
            self.loaded_model = pickle.load(fid)

        with open('./models/count_vectorizer.pkl', 'rb') as vd:
            self.vectorizer = pickle.load(vd)

    #make prediction given input
    vectorizer = None
    @classmethod
    def predict(self, input):
        self.init_model()

        # using model to make prediction
        pred = self.loaded_model.predict(self.vectorizer.transform([input]))[0]

        if pred == "REAL":
            return 0
        elif pred == "FAKE":
            return 1


@application.route('/')
def home():
    return "Hello World!"


@application.route('/ping', methods=['GET'])
def ping():
    return "Ping Testing"

#route that sends output prdiction given news input
@application.route('/predict', methods=['GET'])
def predict():
    news = request.args.get('news')
    if news is None:
        resp = jsonify({'error': 'no input'})
        resp.status_code = 400
        return resp

    # Do the prediction
    pred = Model.predict(news)
    res = {
        "status_code": 200,
        "pred": pred
    }
    return res


if __name__ == '__main__':
    application.run()
