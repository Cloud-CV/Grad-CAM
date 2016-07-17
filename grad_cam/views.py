from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

from grad_cam.utils import grad_cam_vqa

import grad_cam.constants as constants
import uuid
import os

def home(request, template_name="index.html"):
    return render(request, template_name,)


def vqa(request):
    if request.method == "POST":
        # get the parameters from client side
        input_question = request.POST.get('question', '')
        input_answer = request.POST.get('answer', None)
        image = request.FILES['file']

        random_uuid = uuid.uuid1()
        # handle image upload
        vqa_directory = os.path.join(constants.VQA_CONFIG['image_dir'], str(random_uuid))
        if not os.path.exists(vqa_directory):
            os.makedirs(vqa_directory)
        img_path = os.path.join(vqa_directory, image.name)
        handle_uploaded_file(image, img_path)

        # Run the VQA wrapper  
        response = grad_cam_vqa(str(input_question), str(input_answer), str(img_path), str(vqa_directory+"/"))
        response['input_image'] = str(response['input_image']).replace(settings.BASE_DIR, '')
        response['vqa_gcam'] = str(response['vqa_gcam']).replace(settings.BASE_DIR, '')
        response['vqa_gb'] = str(response['vqa_gb']).replace(settings.BASE_DIR, '')
        response['vqa_gb_gcam'] = str(response['vqa_gb_gcam']).replace(settings.BASE_DIR, '')

        return JsonResponse(response)


def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
