import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import openai

CLARIFAI_API_KEY = st.secrets["CLARIFAI_API_KEY"]
openai.api_key = st.secrets["openai_api_key"]

# Initialize Clarifai channel and stub
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

def process_image_with_clarifai(image_bytes, model_id):
    metadata = (('authorization', 'Key ' + CLARIFAI_API_KEY),)
    user_id = 'salesforce'
    app_id = 'blip'

    response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=image_bytes
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )

    if response.status.code != status_code_pb2.SUCCESS:
        st.error("Failed to process the image.")
        return None

    return response.outputs[0]

# Function to validate text
def validate_text(input_text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Is '{input_text}' valid?",
        max_tokens=1000,
        stop=None,
        temperature=0.7,
    )

    if response.choices:
        answer = response.choices[0].text.strip()
        return answer
    else:
        return "No answer found."

def generate_question_from_caption(caption):
    return f"Is '{caption}' valid?"

def answer_question(question):
    response = openai.Completion.create(
        engine="text-davinci-002", 
        prompt=question,
        max_tokens=1000,  # Adjust the max tokens as needed
        stop=None,  
        temperature=0.7,  # Adjust the temperature parameter for response randomness
    )

    if response.choices:
        answer = response.choices[0].text.strip()
        return answer
    else:
        return "No answer found."

st.title("True Captions: Let's validate your content!")
st.write("Great Power comes Great Responsibility. As we increased the use of online news and social media, sometimes fake news and misinformation can be spread easily. This app helps you to validate your content before you share it with the world!🌎")

# Text input field
input_text = st.text_input("Enter text to validate:")

# Validation button
validate_button = st.button("Validate")

# Upload image from user
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


if validate_button and input_text:
    
    validation_result = validate_text(input_text)

    st.subheader("Validation Result:")
    if "valid" in validation_result.lower():
        st.success(validation_result)
    else:
        st.warning(validation_result)


if uploaded_image is not None:
    
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Process the uploaded image and get predictions
    image_bytes = uploaded_image.read()
    model_id = 'general-english-image-caption-blip' 

    output = process_image_with_clarifai(image_bytes, model_id)

    if output is not None:
        # Extract only the caption text
        image_caption = output.data.text.raw


        # Generate a question based on the caption
        question_image = generate_question_from_caption(image_caption)

        st.subheader("Predicted Image Caption using Clarifai Image Caption Generation Model:")
        st.write(image_caption)
        

         # Answer the generated question
        answer = answer_question(question_image)

        st.subheader("Answer:")

        if "valid" in answer.lower():
            st.success(answer)
        else:
            st.warning(answer)
