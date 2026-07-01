import google.generativeai as genai

genai.configure(api_key="AQ.Ab8RN6Iru-LofjPyf9Qo3IUi4Guje1R2l7ovVbeKfZMEgNzEFA")

for model in genai.list_models():
    print(model.name)