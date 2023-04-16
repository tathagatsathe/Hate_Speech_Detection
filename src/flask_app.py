from flask import Flask, request, render_template
import torch
from src.components.model_trainer import BertClassifier, ModelTrainerConfig
from src.utils import tokenize

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertClassifier()
model.load_state_dict(torch.load(ModelTrainerConfig.trained_model_file_path, map_location=torch.device('cpu')))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    text = [request.form['text']]
    input_id, mask = tokenize(text)
    output = model(input_id, mask)
    pred = output.argmax(dim=1)
    prediction = 'Hatespeech' if pred == 2 else ('Offensive' if pred == 1 else 'Neutral')

    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)