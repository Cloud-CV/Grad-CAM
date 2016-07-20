from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

# from grad_cam.utils import grad_cam_vqa, grad_cam_classification

import grad_cam.constants as constants
import uuid
import os
import random


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
        img_path = os.path.join(vqa_directory, str(image))
        handle_uploaded_file(image, img_path)
        # Run the VQA wrapper  
        response = grad_cam_vqa(str(input_question), str(input_answer), str(img_path), str(vqa_directory+"/"))
        response['input_image'] = str(response['input_image']).replace(settings.BASE_DIR, '')
        response['vqa_gcam'] = str(response['vqa_gcam']).replace(settings.BASE_DIR, '')
        response['vqa_gb'] = str(response['vqa_gb']).replace(settings.BASE_DIR, '')
        response['vqa_gb_gcam'] = str(response['vqa_gb_gcam']).replace(settings.BASE_DIR, '')

        return JsonResponse(response)


def classification(request, template_name="classification/classification.html"):
    if request.method == "POST":
        img_path = request.POST.get('img_path')
        label = request.POST.get('label', 243)
        img_path = os.path.join(BASE_DIR, img_path)
        classification_dir = os.path.join(constants.CLASSIFICATION_CONFIG['image_dir'], str(random_uuid))

        # Run the classification wrapper
        response = grad_cam_classification(str(img_path), int(label), str(classification_dir+"/"))
        response['input_image'] = str(response['input_image']).replace(settings.BASE_DIR, '')
        response['classify_gcam'] = str(response['classify_gcam']).replace(settings.BASE_DIR, '')
        response['classify_gb'] = str(response['classify_gb']).replace(settings.BASE_DIR, '')
        response['classify_gb_gcam'] = str(response['classify_gb_gcam']).replace(settings.BASE_DIR, '')
        return response

    demo_images = get_demo_images(constants.GRAD_CAM_DEMO_IMAGES_PATH)
    return render(request, template_name, {"demo_images": demo_images})


def classification_upload(request):
    if request.method == "POST":
        image = request.FILES['file']
        random_uuid = uuid.uuid1()
        # handle image upload
        classification_dir = os.path.join(constants.CLASSIFICATION_CONFIG['image_dir'], str(random_uuid))

        if not os.path.exists(classification_dir):
            os.makedirs(classification_dir)

        img_path = os.path.join(classification_dir, str(image))
        handle_uploaded_file(image, img_path)
        return JsonResponse({"file_path": img_path.replace(settings.BASE_DIR, '')})    


def vqa_upload(request):
    if request.method == "POST":
        image = request.FILES['file']
        random_uuid = uuid.uuid1()
        # handle image upload
        vqa_dir = os.path.join(constants.VQA_CONFIG['image_dir'], str(random_uuid))

        if not os.path.exists(vqa_dir):
            os.makedirs(vqa_dir)

        img_path = os.path.join(vqa_dir, str(image))
        handle_uploaded_file(image, img_path)
        return JsonResponse({"file_path": img_path.replace(settings.BASE_DIR, '')})    


def captioning_upload(request):
    if request.method == "POST":
        image = request.FILES['file']
        random_uuid = uuid.uuid1()
        # handle image upload
        captioning_dir = os.path.join(constants.CAPTIONING_CONFIG['image_dir'], str(random_uuid))

        if not os.path.exists(captioning_dir):
            os.makedirs(captioning_dir)

        img_path = os.path.join(captioning_dir, str(image))
        handle_uploaded_file(image, img_path)
        return JsonResponse({"file_path": img_path.replace(settings.BASE_DIR, '')})    


def handle_uploaded_file(f, path):
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)




def get_demo_images(demo_images_path):
    try:
        demo_images = [random.choice(next(os.walk(demo_images_path))[2]) for i in range(6)]
        demo_images = [os.path.join(constants.GRAD_CAM_DEMO_IMAGES_PATH, x) for x in demo_images]
    except:
        images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', ]
        demo_images = [os.path.join(settings.STATIC_URL, 'images', x) for x in images]
        return demo_images


# def file_upload(request):
#     if request.method == "POST":
#         image = request.FILES['file']
#         random_uuid = uuid.uuid1()

#         # handle image upload
#         vqa_directory = os.path.join(constants.VQA_CONFIG['image_dir'], str(random_uuid))
#         if not os.path.exists(vqa_directory):
#             os.makedirs(vqa_directory)
#         img_path = os.path.join(vqa_directory, str(image))
#         handle_uploaded_file(image, img_path)
#         return JsonResponse({"uploaded_image_path": img_path.replace(settings.BASE_DIR, '')})


# def home_new(request, template_name="index_new.html"):
#     demo_images_path = constants.GRAD_CAM_DEMO_IMAGES_PATH
#     try:
#         demo_images = [random.choice(next(os.walk(demo_images_path))[2]) for i in range(6)]
#         demo_images = [os.path.join(constants.GRAD_CAM_DEMO_IMAGES_PATH, x) for x in demo_images]
#     except:
#         images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', ]
#         demo_images = [os.path.join(settings.STATIC_URL, 'images', x) for x in images]
#     return render(request, template_name,{"demo_images": demo_images})
