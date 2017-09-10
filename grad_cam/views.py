from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from channels import Group

from grad_cam.sender import grad_cam_classification, grad_cam_vqa, grad_cam_captioning
from grad_cam.utils import log_to_terminal
import grad_cam.constants as constants
import uuid
import os
import random
import traceback
import urllib2
import requests
from urlparse import urlparse
from django.http import HttpResponse


def home(request, template_name="index.html"):
    return render(request, template_name,)


def vqa(request, template_name="vqa/vqa.html"):
    socketid = uuid.uuid4()
    if request.method == "POST":
        # get the parameters from client side
        try:
            socketid = request.POST.get('socketid')
            input_question = request.POST.get('question', '')
            input_answer = request.POST.get('answer', None)
            img_path = request.POST.get('img_path')
            img_path = urllib2.unquote(img_path)

            abs_image_path = settings.BASE_DIR + str(img_path)
            # abs_image_path = os.path.join(settings.BASE_DIR, str(img_path[1:]))
            out_dir = os.path.dirname(abs_image_path)
            # Run the VQA wrapper
            log_to_terminal(socketid, {"terminal": "Starting Visual Question Answering job..."})
            response = grad_cam_vqa(str(input_question), str(input_answer), str(abs_image_path), str(out_dir+"/"), socketid)
        except Exception, err:
            log_to_terminal(socketid, {"terminal": traceback.print_exc()})

    demo_images = get_demo_images(constants.COCO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images, 'socketid': socketid})


def classification(request, template_name="classification/classification.html"):
    socketid = uuid.uuid4()
    if request.method == "POST":
        try:
            img_path = request.POST.get('img_path')
            img_path = urllib2.unquote(img_path)
            label = request.POST.get('label')
            socketid = request.POST.get('socketid')

            abs_image_path = os.path.join(settings.BASE_DIR, str(img_path[1:]))
            out_dir = os.path.dirname(abs_image_path)

            # Run the classification wrapper
            log_to_terminal(socketid, {"terminal": "Starting classification job on VGG_ILSVRC_16_layers.caffemodel"})
            response = grad_cam_classification(str(abs_image_path), int(label), str(out_dir+"/"), socketid)
        except Exception, err:
            log_to_terminal(socketid, {"terminal": traceback.print_exc()})
    demo_images = get_demo_images(constants.COCO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images, 'socketid': socketid})


def captioning(request, template_name="captioning/captioning.html"):
    socketid = uuid.uuid4()
    if request.method == "POST":
        try:
            img_path = request.POST.get('img_path')
            img_path = urllib2.unquote(img_path)
            caption = request.POST.get('caption', '')
            socketid = request.POST.get('socketid')

            abs_image_path = os.path.join(settings.BASE_DIR, str(img_path[1:]))
            out_dir = os.path.dirname(abs_image_path)

            # Run the captioning wrapper
            log_to_terminal(socketid, {"terminal": "Starting Captioning job..."})
            response = grad_cam_captioning(str(abs_image_path), str(caption), str(out_dir+"/"), socketid)
        except Exception, err:
            log_to_terminal(socketid, {"terminal": traceback.print_exc()})

    demo_images = get_demo_images(constants.COCO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images, 'socketid': socketid})


def file_upload(request):
    if request.method == "POST":
        image = request.FILES['file']
        demo_type = request.POST.get("type")

        if demo_type == "vqa":
            dir_type = constants.VQA_CONFIG['image_dir']
        elif demo_type == "classification":
            dir_type = constants.CLASSIFICATION_CONFIG['image_dir']
        elif demo_type == "captioning":
            dir_type = constants.CAPTIONING_CONFIG['image_dir']

        random_uuid = uuid.uuid1()
        # handle image upload
        output_dir = os.path.join(dir_type, str(random_uuid))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img_path = os.path.join(output_dir, str(image))
        handle_uploaded_file(image, img_path)
        return JsonResponse({"file_path": img_path})
    else:
        pass



def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def get_demo_images(demo_images_path):
    try:
        images_list = next(os.walk(demo_images_path))[2]
        demo_images = select_random_six_demo_images(images_list)

        demo_images = [os.path.join(settings.MEDIA_URL, 'coco', 'val2014', x) for x in demo_images]
    except:
        images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', ]
        demo_images = [os.path.join(settings.STATIC_URL, 'images', x) for x in images]
    return demo_images


def select_random_six_demo_images(images_list):
    prefixes = ('classify', 'vqa', 'caption')
    demo_images = [random.choice(images_list) for i in range(6)]
    for i in demo_images[:]:
        if i.startswith(prefixes):
            demo_images = select_random_six_demo_images(images_list)
    return demo_images


def upload_image_using_url(request):
    if request.method == "POST":
        try:
            socketid = request.POST.get('socketid', None)
            image_url = request.POST.get('src', None)
            demo_type = request.POST.get('type')

            if demo_type == "vqa":
                dir_type = constants.VQA_CONFIG['image_dir']
            elif demo_type == "classification":
                dir_type = constants.CLASSIFICATION_CONFIG['image_dir']
            elif demo_type == "captioning":
                dir_type = constants.CAPTIONING_CONFIG['image_dir']

            img_name =  os.path.basename(urlparse(image_url).path)
            response = requests.get(image_url, stream=True)

            if response.status_code == 200:
                random_uuid = uuid.uuid1()
                output_dir = os.path.join(dir_type, str(random_uuid))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                img_path = os.path.join(output_dir, str(img_name))
                with open(os.path.join(output_dir, img_name), 'wb+') as f:
                    f.write(response.content)

                img_path =  "/" + "/".join(img_path.split('/')[-5:])
                
                return JsonResponse({"file_path": img_path})
            else:
                return HttpResponse("Please Enter the Correct URL.")
        except:
            return HttpResponse("No images matching this url.")
    else:
        return HttpResponse("Invalid request method.")


def captioning_api(request):
    if request.method == "POST":
        try:
            image = request.FILES['image']
            caption = request.POST.get('caption', '')

            abs_image_path = os.path.join(settings.BASE_DIR, str(img_path[1:]))
            out_dir = os.path.dirname(abs_image_path)

            # Run the captioning wrapper
            log_to_terminal(socketid, {"terminal": "Starting Captioning job..."})
            response = grad_cam_captioning(str(abs_image_path), str(caption), str(out_dir+"/"), socketid)
        except Exception, err:
            log_to_terminal(socketid, {"terminal": traceback.print_exc()})

    demo_images = get_demo_images(constants.COCO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images, 'socketid': socketid})
