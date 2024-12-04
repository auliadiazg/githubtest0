import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import base64
from requests import post, get
import json
import csv
from sklearn import preprocessing

client_id = 'e8b793ed9c9b4b87b4fbe739c978a568' ##ganti variabel dengan client_id milik anda
client_secret = '5caf70cdb38d4281b8b4398866c0c4b9' ##ganti variabel dengan client_secret milik anda
playlistId = '1dtCMTYzAOzwKXqklxPJNS'

## 37i9dQZF1DXbrUpGvoi3TS - 1(similar sad songs)
## 1dtCMTYzAOzwKXqklxPJNS - 2(old songs, rock, rap)
## 0IN7IWKmIfwlEysGyWUuRg - 3(mix of modern electronic, pop, and rock)

dataset = []
dataset2 = []
dataset3 = []

def getToken():
    # gabungkan client_id dan client_secret
    auth_string = client_id + ':' + client_secret

    # encode ke base64
    auth_b64 = base64.b64encode(auth_string.encode('utf-8'))

    # url untuk mengambil token
    url = 'https://accounts.spotify.com/api/token'

    # header untuk mengambil token - sesuai dengan guide dari spotify
    headers = {
        'Authorization': 'Basic ' + auth_b64.decode('utf-8'),
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    # data untuk mengambil token - sesuai dengan guide dari spotify
    data = {'grant_type': 'client_credentials'}

    # kirim request POST ke spotify
    result = post(url, headers=headers, data=data)

    # parse response ke json
    json_result = json.loads(result.content)
    token = json_result['access_token']

    # ambil token untuk akses API
    return token

## panggil fungsi getToken() dibawah ini

## pengambilan token untuk otorisasi API
def getAuthHeader(token):
    return {'Authorization': 'Bearer ' + token}

## pengambilan audio features dari track (lagu)
def getAudioFeatures(token, trackId):
    # endpoint untuk mendapatkan audio features
    url = f'https://api.spotify.com/v1/audio-features/{trackId}'
    headers = getAuthHeader(token)
    result = get(url, headers=headers)  # kirim request GET ke Spotify
    json_result = json.loads(result.content)  # parse response ke JSON

    # Periksa apakah data tersedia
    if json_result and 'danceability' in json_result:
        audio_features_temp = [
            json_result.get('danceability', 0),
            json_result.get('energy', 0),
            json_result.get('key', 0),
            json_result.get('loudness', 0),
            json_result.get('mode', 0),
            json_result.get('speechiness', 0),
            json_result.get('acousticness', 0),
            json_result.get('instrumentalness', 0),
            json_result.get('liveness', 0),
            json_result.get('valence', 0),
            json_result.get('tempo', 0),
        ]
    else:
        # Jika data tidak ditemukan, isi dengan nilai default (misalnya 0)
        audio_features_temp = [0] * 11

    dataset2.append(audio_features_temp)

# pengambilan track (lagu) dari playlist
def getPlaylistItems(token, playlistId):
    # endpoint untuk akses playlist
    url = f'https://api.spotify.com/v1/playlists/{playlistId}/tracks'
    limit = '&limit=100'  # batas maksimal track yang diambil
    market = '?market=ID'  # negara yang tempat aplikasi diakses
    # format data dari track yang diambil
    fields = '&fields=items%28track%28id%2Cname%2Cartists%2Cpopularity%2C+duration_ms%2C+album%28release_date%29%29%29'
    url = url+market+fields+limit  # gabungkan semua parameter
    # ambil token untuk otorisasi, gunakan sebagai header
    headers = getAuthHeader(token)
    result = get(url, headers=headers)  # kirim request GET ke spotify
    json_result = json.loads(result.content)  # parse response ke json
    # print(json_result)

    # ambil data yang diperlukan dari response
    for i in range(len(json_result['items'])):
        playlist_items_temp = []
        playlist_items_temp.append(json_result['items'][i]['track']['id'])
        playlist_items_temp.append(
            json_result['items'][i]['track']['name'].encode('utf-8'))
        playlist_items_temp.append(
            json_result['items'][i]['track']['artists'][0]['name'].encode('utf-8'))
        playlist_items_temp.append(
            json_result['items'][i]['track']['popularity'])
        playlist_items_temp.append(
            json_result['items'][i]['track']['duration_ms'])
        playlist_items_temp.append(
            int(json_result['items'][i]['track']['album']['release_date'][0:4]))
        dataset.append(playlist_items_temp)

    # ambil audio features dari semua track di dalam playlist
    for i in range(len(dataset)):
        getAudioFeatures(token, dataset[i][0])

    # gabungkan dataset dan dataset2
    for i in range(len(dataset)):
        dataset3.append(dataset[i]+dataset2[i])

    print(dataset3)
    # convert dataset3 into csv
    with open('dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "name", "artist", "popularity", "duration_ms", "year", "danceability", "energy", "key", "loudness", "mode",
                         "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"])
        writer.writerows(dataset3)

token = getToken()
print('access token : '+token)
getPlaylistItems(token, playlistId)

def dataProcessing():
    data = pd.read_csv('dataset.csv')
    data
    st.write("## Preprocessing Result")  # streamlit widget

    data = data[['artist', 'name', 'year', 'popularity', 'key', 'mode', 'duration_ms', 'acousticness',
                'danceability', 'energy', 'instrumentalness', 'loudness', 'liveness', 'speechiness', 'tempo', 'valence']]
    data = data.drop(['mode'], axis=1)
    data['artist'] = data['artist'].map(lambda x: str(x)[2:-1])
    data['name'] = data['name'].map(lambda x: str(x)[2:-1])
    st.write("### Data to be deleted:")
    data[data['name'] == '']
    data = data[data['name'] != '']

    st.write("## Normalization Result")  # streamlit widget
    data2 = data.copy()
    data2 = data2.drop(
        ['artist', 'name', 'year', 'popularity', 'key', 'duration_ms'], axis=1)
    x = data2.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data2 = pd.DataFrame(x_scaled)
    data2.columns = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                     'loudness', 'liveness', 'speechiness', 'tempo', 'valence']
    data2

    st.write("## Dimensionality Reduction with PCA")  # streamlit widget
    pca = PCA(n_components=2)
    pca.fit(data2)
    pca_data = pca.transform(data2)
    pca_df = pd.DataFrame(data=pca_data, columns=['x', 'y'])
    fig = px.scatter(pca_df, x='x', y='y', title='PCA')
    st.plotly_chart(fig)  # output plotly chart using streamlit

    st.write("## Clustering with K-Means")  # streamlit widget
    data2 = list(zip(pca_df['x'], pca_df['y']))
    kmeans = KMeans(n_init=10, max_iter=1000).fit(data2)
    fig = px.scatter(pca_df, x='x', y='y', color=kmeans.labels_,
                     color_continuous_scale='rainbow', hover_data=[data.artist, data.name])
    st.plotly_chart(fig)  # output plotly chart using streamlit

    st.write("Enjoy!")


# streamlit widgets
st.write("# Spotify Playlist Clustering")
client_id = st.text_input("Enter Client ID")
client_secret = st.text_input("Enter Client Secret")
playlistId = st.text_input("Enter Playlist ID")

# streamlit widgets
if st.button('Create Dataset!'):
    token = getToken()
    getPlaylistItems(token, playlistId)