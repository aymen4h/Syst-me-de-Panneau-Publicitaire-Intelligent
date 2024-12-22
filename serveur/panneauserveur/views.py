import os
from django.conf import settings
# from django.conf.urls.static import static
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
# from rest_framework.parsers import MultiPartParser, FormParser
import mimetypes
import time
import cv2
import numpy as np
from urllib.parse import unquote
import base64
from deepface import DeepFace
import skfuzzy as fuzz
import random


# Configurer les paramètres Django pour les fichiers statiques et médias
MEDIA_ROOT = os.path.join(settings.BASE_DIR, 'media')
MEDIA_URL = '/media/'

# Assurez-vous que le dossier "videos" existe
VIDEOS_DIR = os.path.join(MEDIA_ROOT, 'videos')
os.makedirs(VIDEOS_DIR, exist_ok=True)


# Vue pour l'upload de fichiers vidéo
@csrf_exempt
@api_view(['POST'])
def upload_video(request):
    if request.method == 'POST':
        try:
            video = request.FILES['video']
            regle = request.data.get('regle')
            # Vérifier le type MIME
            if video.content_type != 'video/mp4':
                return Response({"error": "Only .mp4 files are allowed!"}, status=status.HTTP_400_BAD_REQUEST)

            # Sauvegarder le fichier
            fs = FileSystemStorage(location=VIDEOS_DIR)
            # filename = fs.save(f"{int(time.time())}={video.name}", video)
            filename = fs.save(f"{regle}={int(time.time())}", video)
            file_path = fs.url(filename)

            return Response({"message": "Video uploaded and saved successfully.", "file_path": file_path},
                            status=status.HTTP_201_CREATED)

        except KeyError:
            return Response({"error": "No video file uploaded."}, status=status.HTTP_400_BAD_REQUEST)


# Vue pour récupérer une vidéo
@csrf_exempt
@api_view(['POST'])
def get_video(request):
    param_image = request.data.get('image', '')
    # decoded_param = unquote(param_image)
    decoded_data = base64.b64decode(param_image)  # decoder param du l'url
    np_data = np.frombuffer(decoded_data, np.uint8)  # convertir en tableau numpy
    image_cv2 = cv2.imdecode(np_data, cv2.IMREAD_COLOR)  # lire l'image avec cv2
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    param_rain = request.data.get('rain', '')
    nb_male = 0
    nb_jeune = 0
    nb_adulte = 0
    nb_vieu = 0
    # appliquer deepface
    persons = []
    try:
        persons = DeepFace.analyze(img_path=image_rgb, actions=['age', 'gender'], detector_backend="mtcnn", )
        for pe in persons:
            print(f'\n age:{pe["age"]}, gender: {pe["dominant_gender"]}')
        for person in persons:
            if person["age"] < 30:
                nb_jeune += 1
            elif person["age"] < 50:
                nb_adulte += 1
            else:
                nb_vieu += 1
            if person["dominant_gender"] == "Man":
                nb_male += 1
    except Exception as e:
        print(str(e))

    try:
        nb_male = 100 * nb_male / len(persons)
        nb_jeune = 100 * nb_jeune / len(persons)
        nb_adulte = 100 * nb_adulte / len(persons)
        nb_vieu = 100 * nb_vieu / len(persons)
    except Exception as e:
        print("hadi exception division ", str(e))
    prop_rain = 100 * float(param_rain)
    video_files = os.listdir(VIDEOS_DIR)
    print(f"male: {nb_male}, jeune; {nb_jeune}, adulte: {nb_adulte}, vieu: {nb_vieu}, rain: {prop_rain}")
    try:
        if not video_files:
            return Response({"error": "No videos available."}, status=status.HTTP_404_NOT_FOUND)

        # appliquer logique floue
        video_file_name = selection_floue(video_files, nb_male, nb_jeune, nb_adulte, nb_vieu, prop_rain)
        print("videonnn", video_file_name)
        video_path = os.path.join(VIDEOS_DIR, video_file_name)
        mime_type, _ = mimetypes.guess_type(video_path)

        if mime_type is None:
            mime_type = 'application/octet-stream'

        response = FileResponse(open(video_path, 'rb'), content_type=mime_type)
        response['Content-Disposition'] = 'attachment; filename="video.mp4"'
        return response

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def selection_floue(list_videos, prop_m, prop_j, prop_a, prop_v, prop_r):
    random.shuffle(list_videos)
    x_male = np.arange(0, 101, 5)
    x_jeune = np.arange(0, 101, 5)
    x_adulte = np.arange(0, 101, 5)
    x_vieu = np.arange(0, 101, 5)
    x_rain = np.arange(0, 101, 5)
    x_video = np.arange(0, len(list_videos) * 10, 1)

    prop_male = fuzz.trimf(x_male, [0, 100, 150])
    prop_female = fuzz.trimf(x_male, [0, 0, 100])
    prop_jeune = fuzz.trimf(x_jeune, [0, 100, 150])
    prop_adulte = fuzz.trimf(x_adulte, [0, 100, 150])
    prop_vieu = fuzz.trimf(x_vieu, [0, 100, 150])
    prop_rain = fuzz.trimf(x_rain, [0, 100, 150])
    prop_sun = fuzz.trimf(x_rain, [0, 0, 100])
    prop_video = []
    for i, x in enumerate(list_videos):
        prop_video.append(fuzz.trapmf(x_video, [i * 10, i * 10 + 1, (i + 1) * 10 - 1, (i + 1) * 10]))

    gender_male_val = fuzz.interp_membership(x_male, prop_male, prop_m)
    gender_female_val = fuzz.interp_membership(x_male, prop_female, prop_m)
    jeune_val = fuzz.interp_membership(x_jeune, prop_jeune, prop_j)
    adulte_val = fuzz.interp_membership(x_adulte, prop_adulte, prop_a)
    vieu_val = fuzz.interp_membership(x_vieu, prop_vieu, prop_v)
    rain_val = fuzz.interp_membership(x_rain, prop_rain, prop_r)
    sun_val = fuzz.interp_membership(x_rain, prop_sun, prop_r)
    mapping_ensembles = {
        'm': gender_male_val,
        'f': gender_female_val,
        'j': jeune_val,
        'a': adulte_val,
        'v': vieu_val,
        'r': rain_val,
        's': sun_val,
    }
    regles = []
    for regle in list_videos:
        # Diviser la chaîne en conditions et conclusion
        conditions, conclusion = regle.split('=')

        rul = mapping_ensembles[conditions[0]]
        # Parcourir les conditions restantes
        for i in range(1, len(conditions), 2):
            operator = conditions[i]
            next_condition = conditions[i + 1]
            if operator == '&':  # Opérateur logique ET
                rul = np.fmin(rul, mapping_ensembles[next_condition])
            elif operator == '0':  # Opérateur logique OU
                rul = np.fmax(rul, mapping_ensembles[next_condition])
            else:
                raise ValueError(f"Opérateur non reconnu : {operator}")
        # Ajouter la règle
        regles.append(rul)

    aggregated = np.maximum.reduce([regles[i] * prop_video[i] for i in range(len(prop_video))])
    video_output = fuzz.defuzz(x_video, aggregated, 'lom')
    video_output //= 10

    return list_videos[int(video_output)]
