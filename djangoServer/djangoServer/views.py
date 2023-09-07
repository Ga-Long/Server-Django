from io import BytesIO

from PIL import Image
from django.http import HttpResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt

import requests
import traceback
import os

def get_spectrogram(request):
    if request.method == 'GET':
        # 클라이언트에서 전송한 파일을 가져옵니다.

        audio_path = 'djangoServer/audio/m.m4a'
        url = 'http://localhost:8000/process_audio/'
        try:
            with open(audio_path, 'rb') as f:
                # audio_file = {'m4a' : f}
                if not f.closed:
                    print(f"{audio_path} is opened.")
                spectrogram = requests.post(url, files={'m4a': f})
                # print(spectrogram)

        except FileNotFoundError:
            # 파일이 존재하지 않을 때의 예외 처리
            print(f"{audio_path} does not exist.")
        except Exception as e:
            # 그 외의 예외 처리
            print(f"An error occurred while opening {audio_path}: {e}")

        # 스펙트로그램 이미지를 응답으로 반환합니다.
        response = HttpResponse(spectrogram, content_type='image/jpeg')
        return response

# POST 응답 처리
from pydub import AudioSegment
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from django.conf import settings

FIG_SIZE = (15, 10)
DATA_NUM = 30

# m4a -> wav -> spectrogram / -> model -> result
@csrf_exempt
def process_audio(request):
    global peekIndex, image_url

    print("process_audio")
    try:
        if request.method == 'POST':
            print("POST")
            # POST 요청에서 이미지 파일을 가져옵니다.
            m4a_file = request.FILES['m4a']

            # 소리 + 묵음
            # load the audio files
            audio1 = AudioSegment.from_file(m4a_file, format="m4a")
            audio2 = AudioSegment.from_file("djangoServer/slienceSound.m4a", format="m4a")

            # concatenate the audio files
            combined_audio = audio1 + audio2

            # export the concatenated audio as a new file
            file_handle = combined_audio.export("combined.wav", format="wav")

            # paths.append(file_path)
            sig, sr = librosa.load(file_handle, sr=22050)

            # 에너지 평균 구하기
            sum = 0
            for i in range(0, sig.shape[0]):
                sum += sig[i] ** 2
            mean = sum / sig.shape[0]

            # 피크인덱스 찾기
            for i in range(0, sig.shape[0]):
                if (sig[i] ** 2 >= mean):
                    peekIndex = i
                    break

            START_LEN = 1102
            END_LEN = 20948
            if peekIndex > 1102:
                print(peekIndex)
                startPoint = peekIndex - START_LEN
                endPoint = peekIndex + 22050
            else:
                print(peekIndex)
                startPoint = peekIndex
                endPoint = peekIndex + END_LEN

            # 단순 푸리에 변환 -> Specturm
            fft = np.fft.fft(sig[startPoint:endPoint])
            # 복소공간 값 절댓갑 취해서, magnitude 구하기
            magnitude = np.abs(fft)
            # Frequency 값 만들기
            f = np.linspace(0, sr, len(magnitude))
            # 푸리에 변환을 통과한 specturm은 대칭구조로 나와서 high frequency 부분 절반을 날려고 앞쪽 절반만 사용한다.
            left_spectrum = magnitude[:int(len(magnitude) / 2)]
            left_f = f[:int(len(magnitude) / 2)]
            # STFT -> Spectrogram
            hop_length = 512  # 전체 frame 수
            n_fft = 2048  # frame 하나당 sample 수
            # calculate duration hop length and window in seconds
            hop_length_duration = float(hop_length) / sr
            n_fft_duration = float(n_fft) / sr

            # STFT
            stft = librosa.stft(sig[startPoint:endPoint], n_fft=n_fft, hop_length=hop_length)
            # 복소공간 값 절댓값 취하기
            magnitude = np.abs(stft)
            # magnitude > Decibels
            log_spectrogram = librosa.amplitude_to_db(magnitude)
            FIG_SIZE = (10, 10)
            # display spectrogram
            plt.figure(figsize=FIG_SIZE)
            librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, cmap='magma')

            # matplotlib 라이브러리를 사용하여 생성된 spectrogram 이미지를 jpg 형식으로 저장
            image_path = 'static/images/' + 'test.jpg'

            # save spectrogram image
            # plt.savefig('static/images/' + file_handle[:name_end_pos] + '.jpg')
            # spectrogram 이미지 저장
            plt.savefig(image_path)

            plt.close()

            # 이미지 열기
            image = Image.open(image_path)
            # 저장된 이미지를 열어서 확인
            # os.system('open ' + image_path)  # Mac OS 기준

            # 이미지를 바이트 형태로 변환하여 메모리에 저장
            image_bytes = BytesIO()
            image.save(image_bytes, format='JPEG')
            image_bytes = image_bytes.getvalue()

            # 이미지를 HttpResponse 객체에 첨부 파일로 반환
            response = HttpResponse(image_bytes, content_type='image/jpeg')
            response['Content-Disposition'] = 'inline; filename="spectrogram.jpeg"'
            return response

    except Exception as e:
        print(traceback.format_exc())  # 예외 발생시 traceback 메시지 출력
        return HttpResponseServerError()  # 500 Internal Server Error 응답 반환