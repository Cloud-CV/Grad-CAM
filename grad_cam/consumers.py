from django.http import HttpResponse
from channels.handler import AsgiHandler
from grad_cam.sender import classification
from channels import Group
from grad_cam.utils import log_to_terminal
import json

def ws_connect(message):
    print "User connnected via Socket"


def ws_message(message):
    print "Message recieved from client side and the content is ", message.content['text']
    socketid = message.content['text']
    Group(message.content['text']).add(message.reply_channel)
    log_to_terminal(socketid, {"info": "User added to the Channel Group"})
