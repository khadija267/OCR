from django.shortcuts import render

# Create your views here.
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf

# Load the model
model = tf.saved_model.load('/content/drive/MyDrive/OCR/optimized_model')
def preprocess_image_predict(image_path):

    '''
    Input: image path
    Output: preprocessed img
    '''
    # make sure all images have the same size
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (500, 308))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    adjusted_image = cv2.convertScaleAbs(gray_image, alpha=2.2, beta=1)
    denoised_image = cv2.fastNlMeansDenoising(adjusted_image, None, h=1, templateWindowSize=7, searchWindowSize=1)
    _, binary_image = cv2.threshold(denoised_image, 210, 255, cv2.THRESH_BINARY)
    binary_image_dim = np.expand_dims(binary_image, axis=-1)  # Add extra dimension for batch size
    return binary_image_dim

 
sequences, tokenizer = text_loader(filtered_text_list)


def decode_predictions(predictions, tokenizer):
    # Reverse the tokenizer's word index
    reverse_word_index = dict((i, char) for char, i in tokenizer.word_index.items())

    # Convert the predictions from one-hot encoding to integers
    prediction_integers = np.argmax(predictions, axis=-1)

    # Initialize an empty list to hold the decoded predictions
    decoded_predictions = []

    # Iterate over the sequences in the prediction
    for sequence in prediction_integers:
        # Convert each integer in the sequence to its corresponding character
        # and join them together to form a string
        decoded_sequence = []
        for i in sequence:
            char = reverse_word_index.get(i)
            if char is None:
                continue
            else:
                decoded_sequence.append(char)
        decoded_predictions.append(''.join(decoded_sequence))

    return decoded_predictions
# Inference function
def prepare_predict(model, image, tokenizer):
    # Preprocess the image
    preprocessed_image = preprocess_image_predict(image)

    # Make a prediction
    predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))

    # Decode the predictions
    decoded_predictions = decode_predictions(predictions, tokenizer)

    return decoded_predictions

@csrf_exempt 
def predict(request):
    if request.method == 'POST':
        # Assume the client will send image data in the POST body
        image = Image.open(request.FILES['image'].file)
        
        # Inference
        predictions = predict(model, image, tokenizer)

        # Postprocess the predictions if necessary
        # Here, we simply convert the predictions to a list so they can be JSON serialized
        predictions = predictions.numpy().tolist()

        return JsonResponse({
            'predictions': predictions,
        })
    else:
        return JsonResponse({'error': 'only POST requests are allowed'}, status=405)