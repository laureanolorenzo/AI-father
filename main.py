from flask import Flask, render_template, request, jsonify, url_for
from chatv2 import get_relevant_docs
import json
# from chatv2 import get_response

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

@app.get('/')
def get_index():
    return render_template('index.html')

@app.post('/predict')
def post_predict():
    text = request.get_json().get('message') #Comes from the front end
    # Validation
    # answer = get_response(text) >> From chatv2~
    passages = get_relevant_docs(text)
    if not passages:
        return jsonify({'status':201,'relevant_documents':[]}) #No match!
    # print(passages)
    passages = json.dumps(passages,ensure_ascii=False,check_circular=True)
    # print(passages)
    status = 200
        # status = 500
        # passages = None
    response = {
        'status':status,
        'message': f'You sent: {text}',
        'relevant_documents': passages, #etc
    }
    # print(response)
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)