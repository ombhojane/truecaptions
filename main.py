import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import openai

# Set your Clarifai API credentials
CLARIFAI_API_KEY = 'a672209162e641af9deb4ba82b8606c6'
openai.api_key = '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'

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
        max_tokens=50,
        stop=None,
        temperature=0.7,
    )

    if response.choices:
        answer = response.choices[0].text.strip()
        return answer
    else:
        return "No answer found."

def generate_question_from_caption(caption):
    # You can customize the question format here
    return f"Is '{caption}' valid?"

def answer_question(question):
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can choose a different engine based on your needs
        prompt=question,
        max_tokens=50,  # Adjust the max tokens as needed
        stop=None,  # Add any custom stop words if necessary
        temperature=0.7,  # Adjust the temperature parameter for response randomness
    )

    if response.choices:
        answer = response.choices[0].text.strip()
        return answer
    else:
        return "No answer found."


# Streamlit app title
st.title("AI, Is This Caption Valid?")

# Text input field
input_text = st.text_input("Enter text to validate:")

# Validation button
validate_button = st.button("Validate")

# Upload image from user
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


if validate_button and input_text:
    # Validate the input text
    validation_result = validate_text(input_text)

    st.subheader("Validation Result:")
    st.write(validation_result)


if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Process the uploaded image and get predictions
    image_bytes = uploaded_image.read()
    model_id = 'general-english-image-caption-blip'  # Change this to your desired model ID

    output = process_image_with_clarifai(image_bytes, model_id)

    if output is not None:
        # Extract only the caption text
        image_caption = output.data.text.raw


        # Generate a question based on the caption
        question_image = generate_question_from_caption(image_caption)

        st.subheader("Predicted Image Caption using Clarifai Image Caption Generator model:")
        st.write(image_caption)
        

         # Answer the generated question
        answer = answer_question(question_image)

        st.subheader("Answer:")
        st.write(answer)