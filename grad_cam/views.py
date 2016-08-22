from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from channels import Group

from grad_cam.sender import classification as classify_job
from grad_cam.utils import log_to_terminal

import grad_cam.constants as constants
import uuid
import os
import random

def home(request, template_name="index.html"):
    return render(request, template_name,)


def vqa(request, template_name="vqa/vqa.html"):
    if request.method == "POST":
        # get the parameters from client side
        try:
            demo_type = request.POST.get("demo_type")

            # Deprecated code. Remove it after demo.
            input_question = request.POST.get('question', '')
            input_answer = request.POST.get('answer', None)
            img_path = request.POST.get('img_path')
            # handle image upload
            abs_image_path = os.path.join(settings.BASE_DIR, str(img_path[1:]))
            out_dir = os.path.dirname(abs_image_path)

            # Run the VQA wrapper
            response = grad_cam_vqa(str(input_question), str(input_answer), str(abs_image_path), str(out_dir+"/"))
            response['input_image'] = str(response['input_image']).replace(settings.BASE_DIR, '')
            response['vqa_gcam'] = str(response['vqa_gcam']).replace(settings.BASE_DIR, '')
            response['vqa_gcam_raw'] = str(response['vqa_gcam_raw']).replace(settings.BASE_DIR, '')
            response['vqa_gb'] = str(response['vqa_gb']).replace(settings.BASE_DIR, '')
            response['vqa_gb_gcam'] = str(response['vqa_gb_gcam']).replace(settings.BASE_DIR, '')
            return JsonResponse(response)
        except Exception as e:
            return JsonResponse({"error": str(e)})

    demo_images = get_demo_images(constants.GRAD_CAM_DEMO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images})


def classification(request, template_name="classification/classification.html"):
    if request.method == "POST":
        try:
            img_path = request.POST.get('img_path')
            label = request.POST.get('label')
            socketid = request.POST.get('csrfmiddlewaretoken')

            abs_image_path = os.path.join(settings.BASE_DIR, str(img_path[1:]))
            out_dir = os.path.dirname(abs_image_path)

            # Run the classification wrapper
            print "Debug statement"
            print "abs_image_path", abs_image_path
            print "out_dir", out_dir
            log_to_terminal(socketid, {"terminal": "Starting classification job on VGG_ILSVRC_16_layers.caffemodel"})
            response = classify_job(str(abs_image_path), int(label), str(out_dir+"/"), socketid)
        except Exception as e:
            print str(e)
            return JsonResponse({"error": str(e)})
    demo_images = get_demo_images(constants.GRAD_CAM_DEMO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images})


def captioning(request, template_name="captioning/captioning.html"):
    if request.method == "POST":
        try:
            img_path = request.POST.get('img_path')
            caption = request.POST.get('caption', '')

            abs_image_path = os.path.join(settings.BASE_DIR, str(img_path[1:]))
            out_dir = os.path.dirname(abs_image_path)

            # Run the captioning wrapper
            response = grad_cam_captioning(str(abs_image_path), str(caption), str(out_dir+"/"))
            response['input_image'] = str(response['input_image']).replace(settings.BASE_DIR, '')
            response['captioning_gcam'] = str(response['captioning_gcam']).replace(settings.BASE_DIR, '')
            response['captioning_gcam_raw'] = str(response['captioning_gcam_raw']).replace(settings.BASE_DIR, '')
            response['captioning_gb'] = str(response['captioning_gb']).replace(settings.BASE_DIR, '')
            response['captioning_gb_gcam'] = str(response['captioning_gb_gcam']).replace(settings.BASE_DIR, '')
            return JsonResponse(response)
        except Exception as e:
            return JsonResponse({"error": str(e)})

    demo_images = get_demo_images(constants.GRAD_CAM_DEMO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images})


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
        return JsonResponse({"file_path": img_path.replace(settings.BASE_DIR, '')})    


def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def get_demo_images(demo_images_path):
    try:
        demo_images = [random.choice(next(os.walk(demo_images_path))[2]) for i in range(6)]
        demo_images = [os.path.join(constants.COCO_IMAGES_PATH, x) for x in demo_images]
    except:
        images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', ]
        demo_images = [os.path.join(settings.STATIC_URL, 'images', x) for x in images]

    return demo_images

