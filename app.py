import json
import os, shutil
import sys

import base64
import io

import boto3

from langchain.llms.bedrock import Bedrock

from PIL import Image

## Set up the environment and initialise model & parameter settings

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

llm = Bedrock(
model_id="amazon.titan-tg1-large",
model_kwargs={
    "maxTokenCount": 4096,
    "stopSequences": [],
    "temperature": 0,
    "topP": 1,
},
client=boto3_bedrock,
)

def cleanup_storyimages():
    folder = './storyImages'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Streamlit Application
import streamlit as st

st.title('📖🔨StoryCraft')

# Retrive the role, age, story_flavor, story and length from the user in 5 small input boxes that are side by side
with st.form(key='columns_in_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
        role = st.text_input('Role', value="parent")
    with c2:
        subject = st.text_input('Subject', value="child")
    with c3:
        age = st.text_input('Age', value="6")
    c4, c5, c6 = st.columns(3)
    with c4:
        story_flavor = st.text_input('Story Flavour', value="Funny")
    with c5:
        length = st.text_input('Length', value="200")
    with c6:
        classification = st.selectbox(
                                "Classification",
                                ("Fiction", "Non-Fiction")
                            )
    story = st.text_input('Story', value="MineCraft")

    submitButton = st.form_submit_button(label = 'Craft!')

# If form is submitted execute the following code
if submitButton:
    
    if classification == "Fiction":
        temperature = 0
    else:
        temperature = 1

    parameters={
            "maxTokenCount":3000,
            "stopSequences":[],
            "temperature":temperature,
            "topP":0.9
            }
    prompt_data = "I am a " + role +"." + " I have a " + subject + " who is " + age + " years old. Could you please generate a " + story_flavor + " story for them? The story is about " + story + ". Please write less than " + length + " letters."
    body = json.dumps({"inputText": prompt_data, "textGenerationConfig": parameters})
    modelId = "amazon.titan-tg1-large"  # change this to use a different version from the model provider
    accept = "application/json"
    contentType = "application/json"

    response = boto3_bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    text_answer = response_body.get("results")[0].get("outputText")

    concise_text=[]
    text_splitter=text_answer.split("\n")
    i=0
    for each_sentence in text_splitter:
        if len(each_sentence) != 0:
            concise_text.append(each_sentence)
            i=i+1

    images=()
    images_list=[]
    negative_prompts = [
        "poorly rendered",
        "ugly",
        "poor illustration",
        "deformed",
        "weird"
    ]
    style_preset = "digital-art" #digital-art photographic cartoon 

    i=1
    for each_line in concise_text:
        prompt= f"this story is about {story} " + each_line
        request = json.dumps({
        "text_prompts": (
            [{"text": prompt, "weight": 1.0}]
            + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
        ),
        "cfg_scale": 5,
        "seed": 5450,
        "steps": 30,
        "style_preset": style_preset,
        })
        modelId = "stability.stable-diffusion-xl"

        try:
            response = boto3_bedrock.invoke_model(body=request, modelId=modelId)
            response_body = json.loads(response.get("body").read())
            base_64_img_str = response_body["artifacts"][0].get("base64")
            image_name = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8"))))
            image_name.save(f"./storyImages/image_{i}.png")
            i=i+1
            images_list.append(i)
        except Exception as e:
            print(f"{each_line} ----- Failed due to error: {e}")


    i=1
    for text_line in concise_text:
        try:
            image_to_show=Image.open(f"./storyImages/image_{i}.png")
            # display the image, centered
            st.image(image_to_show, caption=text_line, use_column_width=True)
        except:
            pass
        i=i+1

# Delete button to reset the storyImages
deleteButton = st.button(label = '🔃Reset')

if deleteButton:
    cleanup_storyimages()