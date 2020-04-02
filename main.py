"""
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

import requests

from io import BytesIO
"""

import logging
import uuid
import os
import json
from os.path import join, dirname
from dotenv import load_dotenv
from flask import Flask, request, abort, redirect
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FlexSendMessage, ImageMessage
)
from linepay import LinePayApi
from google.cloud import datastore

# dotenv
load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# logger
logger = logging.getLogger("linepay")
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
logger.addHandler(sh)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
sh.setFormatter(formatter)

# json
json_open = open('data.json', 'r')
json_data = json.load(json_open)

# Messaging API
CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# LINE Pay API
LINE_PAY_CHANNEL_ID = os.environ.get("LINE_PAY_CHANNEL_ID")
LINE_PAY_CHANNEL_SECRET = os.environ.get("LINE_PAY_CHANNEL_SECRET")
LINE_PAY_REQEST_BASE_URL = "https://{}".format(
	# set your server host name (ex. ngrok forwarding host) at HOST_NAME on .env file
	os.environ.get("HOST_NAME")
)
api = LinePayApi(LINE_PAY_CHANNEL_ID, LINE_PAY_CHANNEL_SECRET, is_sandbox=True)

app = Flask(__name__)

# Create Json
def get_plan_json(user_id, json_data):
    return {
  "type": "bubble",
  "size": "micro",
  "header": {
    "type": "box",
    "layout": "vertical",
    "contents": [
      {
        "type": "text",
        "size": "sm",
        "text": json_data["tour"]
      }
    ]
  },
  "hero": {
    "type": "image",
    "url": json_data["tour_image_url"],
    "size": "full",
    "aspectRatio": "20:13",
    "aspectMode": "cover"
  },
  "body": {
    "type": "box",
    "layout": "vertical",
    "contents": [
      {
        "type": "text",
        "text": json_data["price"]
      }
    ]
  },
  "footer": {
    "type": "box",
    "layout": "vertical",
    "contents": [
      {
        "type": "button",
        "style": "primary",
        "action": {
          "type": "uri",
          "label": "LINE Pay 決済",
          "uri": LINE_PAY_REQEST_BASE_URL + "/request?user_id=" + user_id + "&pran_id=" + json_data["id"]
        }
      }
    ]
  }
}

# Create Flex Message
def get_plan_flex_msg(user_id, json_data):
    plan_jsons = []
    for i in range(3):
        plan_jsons.append(get_plan_json(user_id, json_data[i]))
    return {
        "type": "flex",
        "altText": "Flex Message",
        "contents": {
            "type": "carousel",
            "contents": plan_jsons
        }
    }

"""
header = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + CHANNEL_ACCESS_TOKEN
}

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 画像を受け取る部分
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("handle_image:", event)

    line_url = 'https://api.line.me/v2/bot/message/' + event.message.id + '/content/'

    # 画像の取得
    result = requests.get(line_url, headers=header)

    # 画像の保存
    image = Image.open(BytesIO(result.content))

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(str(prediction))

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=str(prediction))
    )
"""

@handler.add(MessageEvent, message=TextMessage)
def message_text(event):
    if event.message.text == "プラン":
		# Flex Message
        flex_obj = FlexSendMessage.new_from_json_dict(get_plan_flex_msg(event.source.user_id, json_data))
        line_bot_api.reply_message(
            event.reply_token,
            messages=flex_obj
        )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=event.message.text)
        )


@app.route("/callback", methods=['POST'])
def callback():
	# get X-Line-Signature header value
	signature = request.headers['X-Line-Signature']

	# get request body as text
	body = request.get_data(as_text=True)
	#app.logger.info("Request body: " + body)

	# handle webhook body
	try:
		handler.handle(body, signature)
	except InvalidSignatureError:
		abort(400)

	return 'OK'

@app.route("/request", methods=['GET'])
def pay_request():
	order_id = str(uuid.uuid4())
	amount = 1
	currency = "JPY"
	# Get QueryParameters
	user_id = request.args.get("user_id")
	if request.args.get("pran_id") == "0":
		product_name = json_data[0]["tour"]
		image_url = json_data[0]["tour_image_url"]
	elif request.args.get("pran_id") == "1":
		product_name = json_data[1]["tour"]
		image_url = json_data[1]["tour_image_url"]
	elif request.args.get("pran_id") == "2":
		product_name = json_data[2]["tour"]
		image_url = json_data[2]["tour_image_url"]
	# Request
	request_options = {
		"amount": amount,
		"currency": currency,
		"orderId": order_id,
		"packages": [
			{
				"id": "package-999",
				"amount": 1,
				"name": "Sample package",
				"products": [
					{
						"id": "product-001",
						"name": product_name,
						"imageUrl": image_url,
						"quantity": 1,
						"price": 1
					}
				]
			}
		],
		"options": {
			"payment": {
				"payType": "PREAPPROVED"
			}
		},
		"redirectUrls": {
			"confirmUrl": LINE_PAY_REQEST_BASE_URL + "/confirm",
			"cancelUrl": LINE_PAY_REQEST_BASE_URL + "/cancel"
		}
	}
	logger.debug(request_options)
	response = api.request(request_options)
	logger.debug(response)
	# Check Payment Satus
	transaction_id = int(response["info"]["transactionId"])
	check_result = api.check_payment_status(transaction_id)
	logger.debug(check_result)
	# Datastore
	client = datastore.Client()
	key = client.key('PayEntity', transaction_id)
	entity = datastore.Entity(key=key)
	entity.update({
        'transaction_id': transaction_id,
		'order_id': order_id,
		'amount': amount,
		'currency': currency,
		'product_name': product_name,
		'user_id': user_id,
	})
	client.put(entity)
	return redirect(response["info"]["paymentUrl"]["web"])


@app.route("/confirm", methods=['GET'])
def pay_confirm():
	transaction_id = int(request.args.get('transactionId'))
	# Datastore
	client = datastore.Client()
	key = client.key('PayEntity', transaction_id)
	entity = client.get(key)
	user_id = entity.get("user_id")
	product_name = entity.get("product_name")
	amount = float(entity.get("amount"))
	currency = entity.get("currency")
	# Confirm
	response = api.confirm(
		transaction_id, 
		amount, 
		currency
	)
	logger.debug(response)
	# push message
	user_id = entity.get("user_id")
	messages = {
        'type': 'flex',
        'altText': 'お支払いを完了しました。',
        'contents': {
        'type': 'bubble',
        'header': {
            'type': 'box',
            'layout': 'vertical',
            'contents': [
            {
                'type': 'text',
                'text': product_name,
                'size': 'md',
                'weight': 'bold'
            },
            {
                'type': 'text',
                'text': 'お支払い完了しました。💰',
                'size': 'md',
                'weight': 'bold'
            },
            {
                'type': 'text',
                'text': 'ありがとうございました。🌟',
                'size': 'md',
                'weight': 'bold'
            }
            ]
        }
        }
    }
	flex_obj = FlexSendMessage.new_from_json_dict(messages)
	line_bot_api.push_message(user_id, messages=flex_obj)
	# Datastore
	client.delete(key)
	
	return "お支払いありがとうございました！"

if __name__ == "__main__":
	app.run(debug=True, port=8000)