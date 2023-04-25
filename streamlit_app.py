import streamlit as st
import torch
from src.components.model_trainer import BertClassifier, ModelTrainerConfig
from src.utils import tokenize

st.set_page_config(page_title="Hate speech detector")
st.title("Hate speech detector")

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier()
    model.load_state_dict(torch.load(ModelTrainerConfig.trained_model_file_path, map_location=device))
    return model

model = load_model()
text = st.text_input('Input text: ','')
if st.button('Submit'):
    mask, input_id = tokenize(text)
    output = model(input_id, mask)
    pred = output.argmax(dim=1)
    prediction = 'Hatespeech' if pred == 1 else 'Neutral'
    st.write(prediction)
