from fastT5 import get_onnx_model, get_onnx_runtime_sessions, OnnxT5
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from pathlib import Path
import os
import random


def load_model(model_path):
    model_name = Path('flan-t5').stem

    # load the model and tokenizer
    encoder_path = os.path.join(model_path, f"{model_name}-encoder-quantized.onnx")
    decoder_path = os.path.join(model_path, f"{model_name}-decoder-quantized.onnx")
    init_decoder_path = os.path.join(model_path, f"{model_name}-init-decoder-quantized.onnx")

    model_paths = encoder_path, decoder_path, init_decoder_path
    model_sessions = get_onnx_runtime_sessions(model_paths)
    model = OnnxT5(model_path, model_sessions)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(
        inputs, 
        num_beams=5, 
        max_length=100,
        no_repeat_ngram_size=3, 
        early_stopping=True,
        top_k = 150,
        top_p = 0.92,
        repetition_penalty = 2.1, 
        num_return_sequences=1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def preprocess(sentence, history):
    hist_size = len(history)
    text = sentence

    if(hist_size != 0 or hist_size < 2):
        text = history[0]+" "+ text
    else:
        text = history[hist_size-2]+" "+ history[hist_size-1]+" "+text

    return text 


def calculate_similarity(sentence1, sentence2):
    # Initialize CountVectorizer with suitable configurations
    vectorizer = CountVectorizer()

    # Create the vectors for sentence1 and sentence2
    vectors = vectorizer.fit_transform([sentence1, sentence2]).toarray()

    # Calculate the cosine similarity between the vectors
    similarity = cosine_similarity([vectors[0]], [vectors[1]])

    return similarity[0][0]

def transition():
    question_list = [
        "What is your hobby?",
		"What do you like to do in your free time?",
		"What are you doing today?",
		"Do you know the famous coffee shop in this city?",
		"Do you prefer tea or coffee?",
		"what kind of music do you like to listen to?",
		"Do you like reading books?",
		"Do you prefer art or sport?",
		"How’s the weather today?",
		"Who is your favorite person in the world and why?",
		"What type of house do you live in?",
		"Do you have any pet?",
		"Do you have any brother or sisters?",
		"Do you like to cook?",
		"Who cooks in your house?",
		"Do you like spicy food?",
		"Which country has the best food that you ever try?",
		"Where was the last place you went on holiday?",
		"Where is the best place you've ever been on holiday?",
		"Do you prefer beach or mountains for holiday destination?",
		"Are you using map or an app while traveling?",
		"Do you like to travel alone or in group?",
		"Do you like to learn new language?",
		"Which place do you want to live in your retired time?",
		"What is your dream when you were a kid?"
    ]

    list_size = len(question_list)
    
    # Initialize the random number generator with a seed
    index = random.randint(0,list_size-1)
     
    return question_list[index]

def post_process(sentence, history):
    # check the similarity
    size = len(history)
    if(size > 2):
        similarity_score = calculate_similarity(sentence, history[size-2])
        if similarity_score >= 0.6:
            # do transition by randomly select the question from the defined list
            text = transition()
            return text
        else:
            return sentence
    else:
        return sentence



def main():
    model_path = "model"
    model, tokenizer = load_model(model_path)

    conv_history = []
    conv_history.append('Hello')
    print("type 'bye' to quit the conversation")
    print("Robot: "+conv_history[0])

    while True:
        user_input = input("You: ")
        if user_input == "bye":
            break
        
        input_text = preprocess(user_input, conv_history)
        conv_history.append(input_text)

        # generate the response
        response = generate_response("generate response: "+input_text, model, tokenizer)
        post_text = post_process(response, conv_history)

        print("Robot: "+post_text)
        conv_history.append(post_text)


if __name__ == "__main__":
    main()