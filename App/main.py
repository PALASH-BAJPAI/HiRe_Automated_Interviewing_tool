### General imports ###
from __future__ import division
from turtle import position
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt


### Flask imports ###
from flask import Flask, render_template, session, request, redirect, flash, Response
import requests


### Score ###
import os 
from pydub.silence import split_on_silence
from playsound import playsound
from pydub import AudioSegment 
from pydub.playback import play 
import pydub
import speech_recognition as sr
GOOGLE_KEY =  "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCyy6eGii1aGFPE\nFVGrDCvsoDXzQU8MSmuJ9OQ/4E3t95sNis5ZoKlUQ0QVrOTUw4kYg1idxzxI+KMu\n5xcW7eu6eLOb0YzZz9tzrRlsHr8TxIsUoDHamGZoQONHojC0BDVW3v5aG6vB+Qyo\nXSq6E5G7AN8oFA6brOT6TpDKwX8tl8K1cvYsn/2Q+q+QsMWizxl22SPImozDZA+2\nQCtLPkAF0tHYUDizQ/2mdP7F+SQFlVnH+K7yzoiiMXl8d8qwbZVFqO0cMJn0Wj6X\nNk/i8zwzr/gewgZ1eHp1MfPpvwrX8FU2s5ggNpAF37gIa+nMFPTbQITRMP/xdqYo\n28ocLyM/AgMBAAECggEARohTkp4g7PlK1kAowwberw62qbs6UVlsWfRrNI2qgHVc\ny/dVlwLruat9iOV3Mj3e7/Ykt71YmVrImSCdubRq+VlTVWVRoL1AT75aGI56h3RR\n/3WApUDYqUjrwB8KAoHkftwiT+65j6BNb3+tctF0fGaIIhnjd2M5w0rKEMpLfvLD\n46Y+YyzHyGUTxmyc0Sc67UFvc00Sh8GtJDj47aAxGUNxNl9n8NHg8tt7vADfYlBS\nVpUF2gf9iv/ddc55PXpH0gp1v5vWrtSXp7DBo6wDDemNpDvuF1L2RAfcBPcZ4Jfs\nJQFM21lEgzHfvu9RbYPBW9ekD8mSIzooJFbGYIa3ZQKBgQD7ZN2vbhZQncn3fnWy\nxPQ3XrvRODieM7k+vfggZkk3QAJMzfETFNSBx9YwXsIM5j2wvv+9oFT9/Bl5DOkl\ns5Jps93MXPP7l0DigUEr4z8swg9d9xvSU0THMRJIwa4yz24hkjCWfRUtqMp8MY3u\nhb0KpqdHfwPguuvkc69n2T1zKwKBgQC2EkYXETUeSnZm/TXE0RxZX4nlNF0P0Y7G\nZUQdbMVmEwJ8My0p7Is4tRKLVR0GSYrgaaL5gCn2v5tmpFhbSbrNf3FAAbzG0P+m\nQiprc7aMZbRPu+P9E4WUHVn0tbyZg+7z1Iqr/67D/0co8sPlbsf5+S1Qb/rUp0SQ\nJL/U428WPQKBgQCE3Ulg53D5yHsux/JSuk9MWFAxgmJCEposM+DI1uaJQdY1W363\nFAJAWSq1w88RXDpsiHXHdc6VscCQvqcWWvLd1Mc7tEDqzoTncWLNXDxOXn4arnhQ\nz9uA30mHlH2JsyHEsmvljVQ9HoFt5A7camiAEZZFbjbRdlkoE5A39ZPJowKBgF0R\nYjqQVTKypWtnq4B7053rtDUxWxCm8fB/+x1/aDgRJ4gNMNzpSREnnd8TFs8L8K1d\n0izvUoQK1YjWIUQooBBDQMSTHsgSNVvrHnvmnj2OD2lihdvrirB3gHASJeHjCtYg\novHgtJkDeIB596Djy9z/fiZL10+0YNu9rUFJMJORAoGAZZ8hYBHzjeBEnA/pV9Wo\n316OxkM+3h33mPy0SQCwvquJjssHAl7Jm2zw/B4W0HJRzLI7fp68ncnW272evs28\nWR/3YiMYcU67p4bbJHLmZ91ppCB8WEVE+7q6AilPudSCkQpR+/oOP3HMduQ17KrC\nqHySdLC82JBAEVwN7KANxRo=\n-----END PRIVATE KEY-----\n"
lang = "pl"


### Audio imports ###
from library.speech_emotion_recognition import *


### Text imports ###
from library.text_emotion_recognition import *
from library.text_preprocessor import *
from nltk import *
import tika
tika.initVM()
from tika import parser
from werkzeug.utils import secure_filename
import tempfile




# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'


### HOME ###

# Home page
@app.route('/')
def home():
    return render_template('home.html')

### INDEX ###

# Index page
@app.route('/platform',methods=['POST','GET'])
def platform():
    return render_template('platform.html')


### ABOUT ###

# About page
@app.route('/about')
def about():
    return render_template('about.html')



### BLOGS ###

# Blogs page
@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/blog_content')
def blogs_content():
    return render_template('blogs_content.html')



################################################################### Interview Scorer ##################################################################################
global name
global duration
global job_position
global text



# Score Home page
@app.route('/score',methods=['POST','GET'])
def score():
    return render_template('scorehome.html')


######### Innterview text #############
@app.route('/interview_text', methods=['POST','GET'])
def interview_text() :
    global duration
    global name
    global job_position
    if request.method == 'POST':
        name=request.form['name']
        age=int(request.form['age'])
        job_position=request.form['position']
        duration=int(request.form['duration'])
    # Flash message
    return render_template('interview_text.html')

def get_personality(text):
    try:
        pred = predict().run(text, model_name = "Personality_traits_NN")
        return pred
    except KeyError:
        return None

def get_text_info(text):
    text = text[0]
    words = wordpunct_tokenize(text)
    common_words = FreqDist(words).most_common(100)
    counts = Counter(words)
    num_words = len(text.split())
    return common_words, num_words, counts

def preprocess_text(text):
    preprocessed_texts = NLTKPreprocessor().transform([text])
    return preprocessed_texts



######### Innterview audio #############
@app.route('/interview',methods=['POST','GET'])
def interview():
    global text
    print(request.form['answer'])
    text =request.form['answer']
    
    flash("After pressing the button above, you will get " + str(duration) +" sec to answer the question.")
    return render_template('interview.html',name=name,display_button=False,color='#C7EEFF')





# Audio Recording
@app.route('/audio_recording_interview', methods=("POST", "GET"))
def audio_recording_interview():
    global duration
    global text
    

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition()

    # Voice Recording

    rec_duration = duration # in sec
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    # Send Flash message
    flash("Recording over! Evaluate your answer on basis of emotions you expressed.")
    
    ############################### Conversion to text #########################
    sound="tmp/voice_recording.wav"
    
    r=sr.Recognizer()
    
    with sr.AudioFile(sound) as source:
        r.adjust_for_ambient_noise(source)
                
        audio=r.listen(source)
        
        try:
            text +=r.recognize_google(audio)
            print(text)
        except Exception as e:
            print(e)
            
    ### text end ###
     
    return render_template('interview.html', display_button=True,name=name,text=text,color='#00ffad')



# Interview Analysis
@app.route('/interview_analysis', methods=("POST", "GET"))
def interview_analysis():
    global text
    global name
    global job_position

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Record sub dir
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')

    # Get most common emotion of other candidates
    df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")

    # Get most common emotion during the interview for other candidates
    major_emotion_other = df_other.EMOTION.mode()[0]

    # Calculate emotion distribution for other candidates
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION==emotion]) / len(df_other)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(os.path.join('static/js/db','audio_emotions_dist_other.txt'), sep=',')


    ############################ for text #############################
    
    print(text)
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    
    df_text = pd.read_csv('static/js/db/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]
    
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])
    
    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]
    
    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.loc[df_mean['Value'].idxmax()]['Trait']
    
    probas = [int(e*100) for e in probas]
    
    data_traits = zip(traits, probas)
    
    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []
    
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    
    trait = traits[probas.index(max(probas))]
    
    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()
    
    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]
    
    # Sleep
    time.sleep(0.5)
    
    ################ SCORE CALCULATION ##########################
    text_model = pickle.load(open('Models/text_score.sav', 'rb'))
    text_data=probas
    t_score=text_model.predict([text_data])[0]
    print(t_score)
    
    audio_model = pickle.load(open('Models/audio_score.sav', 'rb'))
    audio_data=emotion_dist
    a_score=audio_model.predict([audio_data])[0]
    print(a_score)
    
    score = (73.755*a_score + 26.2445*t_score)/100
    
    score=round(score,2)
    print(score)

    

    return render_template('score_analysis.html', a_emo=major_emotion, a_prob=emotion_dist,t_text=text, t_traits = probas, t_trait = trait, t_num_words = num_words, t_common_words = common_words_perso,name=name,position=job_position, score=score)

    







######################################################################## AUDIO INTERVIEW ##################################################################


# Audio Index
@app.route('/audio', methods=['POST' , 'GET'])
def audio_index():

    # Flash message
    flash("After pressing the button above, you will get 15sec to answer the question.")
    
    return render_template('audio.html', display_button=False,color='#C7EEFF' )

# Audio Recording
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition()

    # Voice Recording
    rec_duration = 16 # in sec
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')
    SER.voice_recording(rec_sub_dir, duration=rec_duration)

    # Send Flash message
    flash("Recording over! Evaluate your answer on basis of emotions you expressed.")

    return render_template('audio.html', display_button=True,color='#00ffad')


# Audio Emotion Analysis
@app.route('/audio_analysis', methods=("POST", "GET"))
def audio_analysis():

    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Record sub dir
    rec_sub_dir = os.path.join('tmp','voice_recording.wav')

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')

    # Get most common emotion of other candidates
    df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")

    # Get most common emotion during the interview for other candidates
    major_emotion_other = df_other.EMOTION.mode()[0]

    # Calculate emotion distribution for other candidates
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION==emotion]) / len(df_other)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(os.path.join('static/js/db','audio_emotions_dist_other.txt'), sep=',')

    # Sleep
    time.sleep(0.5)

    return render_template('audio_analysis.html', emo=major_emotion, emo_other=major_emotion_other, prob=emotion_dist, prob_other=emotion_dist_other)





############################ TEXT INTERVIEW ##############################
global df_text

tempdirectory = tempfile.gettempdir()

@app.route('/text', methods=['POST','GET'])
def text() :
    return render_template('text.html')

def get_personality(text):
    try:
        pred = predict().run(text, model_name = "Personality_traits_NN")
        return pred
    except KeyError:
        return None

def get_text_info(text):
    text = text[0]
    words = wordpunct_tokenize(text)
    common_words = FreqDist(words).most_common(100)
    counts = Counter(words)
    num_words = len(text.split())
    return common_words, num_words, counts

def preprocess_text(text):
    preprocessed_texts = NLTKPreprocessor().transform([text])
    return preprocessed_texts

@app.route('/text_analysis', methods=['POST'])
def text_analysis():
    
    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    
    df_text = pd.read_csv('static/js/db/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]
    
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])
    
    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]
    
    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.loc[df_mean['Value'].idxmax()]['Trait']
    
    probas = [int(e*100) for e in probas]
    
    data_traits = zip(traits, probas)
    
    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []
    
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    
    trait = traits[probas.index(max(probas))]
    
    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()
    
    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]
    
    

    return render_template('text_analysis.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)


ALLOWED_EXTENSIONS = set(['pdf'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/text_input', methods=['POST'])
def text_pdf():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    
    text = parser.from_file(f.filename)['content']
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    
    df_text = pd.read_csv('static/js/db/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]
    
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])
    
    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]
    
    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.ix[df_mean['Value'].idxmax()]['Trait']
    
    probas = [int(e*100) for e in probas]
    
    data_traits = zip(traits, probas)
    
    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []
    
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    
    trait = traits[probas.index(max(probas))]
    
    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()
    
    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template('text_dash.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)



### RUN APP ###
if __name__ == '__main__':
    app.run(debug=True)
