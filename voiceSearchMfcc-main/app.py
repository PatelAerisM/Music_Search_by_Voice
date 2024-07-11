from flask import Flask, render_template, request
import io

app = Flask(__name__)

import librosa
import numpy as np
import noisereduce as nr
import scipy.signal
import os
from numpy import dot, linalg
from fastdtw import fastdtw
from scipy.spatial import minkowski_distance
from scipy.spatial.distance import euclidean

def normalize_signal(signal):
    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    return normalized_signal

def preemphasize(y):
    x = []
    i = 0
    for ele in y:
        if i == 0:
            x.append(ele)
        else:
            x.append(ele - 0.97 * y[i-1])
        i += 1
    return x

def extract_mfcc(audio_path, n_mfcc=13, reduce_noise=True, trim_silence=True):
    y, sr = librosa.load(audio_path)
    y = preemphasize(y)
    y = np.array(y)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)
    if reduce_noise:
        y = nr.reduce_noise(y=y, sr=sr)
    sos = scipy.signal.butter(10, 100, 'hp', fs=sr, output='sos')
    filtered_audio = scipy.signal.sosfilt(sos, y)
    y, _ = librosa.effects.trim(filtered_audio)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def dtw_similarity(query_mfcc, database_mfcc):
    distance, path = fastdtw(query_mfcc.T, database_mfcc.T, dist=euclidean)
    return distance

def cosine_similarity(a, b):
    return dot(a, b) / (linalg.norm(a) * linalg.norm(b))

def find_best_match(query_mfcc, target_mfcc, path):
    min_distance = float('inf')
    for start in range(target_mfcc.shape[1] - query_mfcc.shape[1] + 1):
        segment = target_mfcc[:, start:start + query_mfcc.shape[1]]
        distance_arr = minkowski_distance(query_mfcc, segment)
        mean = np.mean(distance_arr)
        distance = sum(value for value in distance_arr if value <= 3 * mean)
        if distance < min_distance:
            min_distance = distance
    return min_distance

def find_similar(query_audio_path, database_features, top_n=5):
    query_mfcc = extract_mfcc(query_audio_path)
    distances = {}

    for path, mfcc in database_features.items():
        if mfcc.shape[1] >= query_mfcc.shape[1]:
            distance = find_best_match(query_mfcc, mfcc, path)
            distances[path] = distance
        else:
            segment_query = query_mfcc[:, 0:mfcc.shape[1]]
            distance_arr = minkowski_distance(segment_query, mfcc)
            mean = np.mean(distance_arr)
            distance = sum(value for value in distance_arr if value <= 3 * mean)
            distances[path] = distance
    sorted_distances = sorted(distances.items(), key=lambda item: item[1])
    return sorted_distances[:top_n]

# Update the folder path
folder_path = r'C:\Users\patel\Downloads\voiceSearchMfcc-main\voiceSearchMfcc-main\Recorded_Voices'
audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

database_features = {}

for file in audio_files:
    file_path = os.path.join(folder_path, file)
    features = extract_mfcc(file_path)
    database_features[file_path] = features

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        query_audio_path = "query_audio.wav"
        uploaded_file.save(query_audio_path)
        similar_clips = find_similar(query_audio_path, database_features, top_n=5)
        return render_template('index.html', similar_clips=similar_clips)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)