# Lambda code with session-based conversation and payload persistence using Redis for Claude via Bedrock

import json
import boto3
import urllib.request
import urllib.error
import traceback
import redis
import os

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

redis_client = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True,
)

MODEL_ID = "anthropic.claude-v2:1"
GETPLAN_URL = "https://ezk1xmweo1.execute-api.ap-south-1.amazonaws.com/Prod/getPlan"

INITIAL_PROMPT = """
You are a helpful travel assistant. Your goal is to collect the required travel details from the user one step at a time in a friendly and conversational manner. You must gather the following information in this order:
1) Greet and ask how you can help
Note: 
1.1 if any of the below required infornation is given as part of the above response, collect and confirm those before proceed to the next details
1.2 Dont prompt anything else other than Greeting message  
2) Start Date of the travel, follow the "Instruction for entering date:" as defined below
3) End date of the travel, follow the "Instruction for entering date:" as defined below
4) Number of Adults travelling
5) Number of children travelling and there respective age
6) Number of infants travelling

Instruction for entering date:
- Just ask when user is planning to start the travel and dont provode any additional information on how to enter the date 
Note the following 
- Do not prompt any formate to guide him.
- If entered day, month and year then generate the DD-MM-YYYY and confirm with user before proceeding to next
- If entered only day and month then assume the current and generate the DD-MM-YYYY and confirm with user before proceeding to next
- If entered only day then assume the current month and year and generate the DD-MM-YYYY and confirm with user before proceeding to next
- Validate the entered day range and ask user to reenter again if its out of valid range, condider the leap year, should be future day
- Validate the entered month and ask user to reenter again if its not a valid month, should be current or future month
- Validate the entered year and ask user to reenter again if its not a valid year, should be current or future year
- After receiving the input from the user, dont provide any justification why you assumed that date, just confirm the generated date by providing Yes/No
- If user says Yes, then remember the date

Instruction for getting the number of children and there age:
3. Number of adults
4. Number of children
5. If children > 0, ask for their ages (comma-separated)
6. Validate formats wherever applicable. Re-prompt clearly if invalid input is received.
7. Confirm all details with the user at the end.
8. Do not assume values — ask the user directly.
9. Keep the tone warm, clear, and professional.
10. After confirmation, respond with a JSON snippet containing the collected details using the following format:
Keep the destination, source and agent_id same as mentioned in the following JSON 

{"destination":{"name":"Goa, India","placeId":"ChIJ2cxhM6nAvzsRYb7lJAsSmN0","lat":"15.4909301","lon":"73.82784959999999"},"destinations":[],"source":{"name":"Bangalore, Karnataka, India","placeId":"ChIJbU60yXAWrjsR4E9-UejD3_g","lat":"12.9715987","lon":"77.5945627"},"startDateTime":"2025-06-03T00:00:00.000+05:30","endDateTime":"2025-06-05T00:00:00.000+05:30","adults":{"count":"1"},"children":{"count":"0","age":[]},"infants":{"count":"0","age":[]},"purpose":"leisure","pagination":{"start":"0","count":"10","sort_by":"cost","order":"asc"},"rooms":"1","roomsOccupancy":[{"adults":{"count":"1"},"children":{"count":"0","age":[]},"infants":{"count":"0","age":[]}}],"Agent":{"agent_id":"BT1"}}
"""

def get_session_data(user_id, session_id):
    key = f"{user_id}:{session_id}"
    try:
        data = redis_client.hgetall(key)
        print(f"[INFO] Retrieved session for user_id={user_id}, session_id={session_id}")
        conversation = data.get("conversation_history")
        payload_raw = data.get("last_payload")
        payload = json.loads(payload_raw) if payload_raw else None
        return (
            conversation
            or f"\n\nHuman: {INITIAL_PROMPT}\n\nAssistant: Hi! I’m your travel assistant. How can I help you today?",
            payload,
        )
    except Exception as e:
        print(f"[ERROR] Failed to get session data: {str(e)}")
        return (
            f"\n\nHuman: {INITIAL_PROMPT}\n\nAssistant: Hi! I’m your travel assistant. How can I help you today?",
            None,
        )

def save_session_data(user_id, session_id, conversation, payload=None):
    key = f"{user_id}:{session_id}"
    try:
        mapping = {"conversation_history": conversation}
        if payload is not None:
            mapping["last_payload"] = json.dumps(payload)
        redis_client.hset(key, mapping=mapping)
        print(f"[INFO] Session saved for user_id={user_id}, session_id={session_id}")
    except Exception as e:
        print(f"[ERROR] Failed to save session data: {str(e)}")

def lambda_handler(event, context):
    try:
        print("[INFO] Lambda invoked with event:", json.dumps(event))

        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)

        user_input = body.get("message", "")
        session_id = body.get("session_id", "default-session")
        user_id = body.get("user_id", "anonymous")

        print(f"[DEBUG] User input: {user_input}, session_id: {session_id}, user_id: {user_id}")

        prior_history, previous_payload = get_session_data(user_id, session_id)
        full_history = prior_history + f"\n\nHuman: {user_input}\n\nAssistant:"
        print("[DEBUG] Full prompt sent to Bedrock:", full_history[-1000:])

        bedrock_response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "prompt": full_history,
                "max_tokens_to_sample": 1024,
                "temperature": 0.7,
                "stop_sequences": ["\n\nHuman:"]
            }),
            accept="application/json",
            contentType="application/json"
        )

        reply = json.loads(bedrock_response["body"].read())["completion"].strip()
        print("[DEBUG] LLM reply:", reply)
        updated_history = full_history + " " + reply

        extracted_payload = None
        try:
            start = reply.index('{')
            end = reply.rindex('}') + 1
            extracted_payload = json.loads(reply[start:end])
            print("[INFO] Extracted payload from reply:", json.dumps(extracted_payload))

            required_keys = ["destination", "source", "startDateTime", "endDateTime", "adults", "children", "infants"]
            if all(k in extracted_payload for k in required_keys):
                save_session_data(user_id, session_id, updated_history, payload=extracted_payload)

                try:
                    print("[INFO] Sending payload to getPlan API")
                    post_data = json.dumps(extracted_payload).encode("utf-8")
                    req = urllib.request.Request(GETPLAN_URL, data=post_data, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req) as res:
                        getplan_response = res.read().decode("utf-8")

                    print("[INFO] getPlan API response:", getplan_response)
                    return {
                        "statusCode": 200,
                        "body": json.dumps({
                            "reply": reply,
                            "getPlanResult": getplan_response,
                            "payload": extracted_payload,
                            "conversation_history": updated_history,
                            "model_input": full_history
                        })
                    }
                except Exception as e:
                    print("[ERROR] getPlan call failed:", str(e))
                    return {
                        "statusCode": 500,
                        "body": json.dumps({"error": f"GetPlan failed: {str(e)}"})
                    }

        except (ValueError, json.JSONDecodeError) as e:
            print(f"[WARNING] No valid JSON payload detected in LLM reply: {str(e)}")

        save_session_data(user_id, session_id, updated_history, payload=extracted_payload or previous_payload)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "reply": reply,
                "payload": extracted_payload or previous_payload,
                "conversation_history": updated_history,
                "model_input": full_history
            })
        }

    except Exception as e:
        print("[ERROR] Unhandled exception:")
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
