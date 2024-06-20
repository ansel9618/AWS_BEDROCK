import json
import boto3

prompt_data = """
is llama 2 70B model trained till data of 2024?
"""
bedrock = boto3.client(service_name="bedrock-runtime")

payload={
        "prompt":"[INST]" + prompt_data+ "[/INST]",
        "max_gen_len": 512,
        "temperature": 0.5
        }

model_id="meta.llama2-70b-chat-v1"
body = json.dumps(payload)
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept = "application/json",
    contentType = "application/json"
)

response_body = json.loads(response.get("body").read())
response_text = response_body['generation']
print(response_text)
