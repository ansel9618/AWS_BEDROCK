import json
import boto3
import os
import base64

prompt_data = """
provide me an 4K hd image of a beach, with a fire tornado and darth vader relaxing
"""
bedrock = boto3.client(service_name="bedrock-runtime")
prompt_template = [{"text":prompt_data,"weight":1}]
payload={
        "text_prompts":prompt_template,
        "cfg_scale":10,
        "seed":0,
        "steps":50,
        "width":1024,
        "height":1024
        }
body = json.dumps(payload)
model_id="stability.stable-diffusion-xl-v1"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept = "application/json",
    contentType = "application/json"
)

response_body = json.loads(response.get("body").read())
print(response_body)

# note the reponse will be key value pair with artifacts
# inside the artifact we'll get a base64 encoded image
# so we'll have to encode it with utf-8 format to read th image
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)


#saving the image in a o/p file directory
output_dir = "output"
os.makedirs(output_dir,exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name,"wb") as f:
    f.write(image_bytes)
