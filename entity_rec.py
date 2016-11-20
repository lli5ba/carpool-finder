import os
from nltk import StanfordNERTagger
import nltk
import pandas as pd
import requests

__author__ = 'Leandra'

stanford_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stanfordjars')
st = StanfordNERTagger('ner-model.ser.gz', os.path.join(stanford_dir, 'stanford-ner.jar'))
st._stanford_jar = os.path.join(stanford_dir, '*')
url_base = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
api_key = 'AIzaSyAVat82-OUFKC9GpyOi3LNyQKwxE2KWY9U'


def load_data(csv_fpath):
        print("loading data")
        data = pd.read_csv(csv_fpath, encoding = 'latin1')

        return data


def processLanguage(messages):
    posts = []
    for (item, _) in messages:
        tokenized = nltk.word_tokenize(item)
        st.tag(tokenized)
    print(posts)
    tagging = st.tag(posts)
    print(tagging)
    print(extract_chunks(tagging))
    print(messages[0][0])
    tokenize = nltk.word_tokenize(messages[0][0])
    print(st.tag(tokenize))

def process_sentence(message):
    tokenized = nltk.word_tokenize(message)
    tagged_sent = st.tag(tokenized)
    locations = extract_chunks(tagged_sent)
    get_coords(locations, '')



def process_all(messages):
    for (message, _) in messages:
        process_sentence(message)

def extract_chunks(tagged_sent, chunk_type='LOC'):
    locations = []
    chain = False
    location = ''
    for (word, tag) in tagged_sent:
        if tag == chunk_type and not chain: #start of chain
            location += word
            chain = True
        elif tag == chunk_type and chain: #add on to chain
            location += " " + word
        elif tag != chunk_type and chain: #chain ended
            locations.append(location)
            location = ''
            chain = False
    return locations

def get_coords(locations, state):
    coordinates = []
    for location in locations:
        #construct uri
        if state != '':
            location += " " + state
        parameters = {'query': location, 'key': api_key}
        r = requests.get(url_base, params=parameters)
        if r.status_code == 200:
            response = r.json()
            if 'results' in response:
                results = response['results']
                #get coordinates
                print results[0]
                if results and 'geometry' in results[0]:
                    geometry = results[0]['geometry']
                    if 'location' in geometry:
                        coordinates.append(geometry['location'])
    print(coordinates)





if __name__ == '__main__':
    csv = "shuffled_posts.csv"
    data = load_data(csv)
    print(data['status_message'][400])
    #process_all(zip(data['status_message'][400], data['status_published'][400]))
    process_sentence('Looking for a ride back to Blacksburg from Fredericksburg/Stafford/Richmond/Chesterfield on Sunday!')