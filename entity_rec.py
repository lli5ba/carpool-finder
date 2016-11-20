import json, os, re, urllib2, datetime
import pickle
from nltk import StanfordNERTagger
import nltk
import pandas as pd
import requests
from sutime import SUTime
from collections import defaultdict
import numpy as np
__author__ = 'Leandra'

debug = False

#location global vars
stanford_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stanfordjars')
st = StanfordNERTagger('ner-model.ser.gz', os.path.join(stanford_dir, 'stanford-ner.jar'))
st._stanford_jar = os.path.join(stanford_dir, '*')
place_to_coords = {}
home_location = ''
url_base = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
api_key = 'AIzaSyAVat82-OUFKC9GpyOi3LNyQKwxE2KWY9U'

#time global vars
jar_files = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sutimejars')
sutime = SUTime(jars=jar_files, mark_time_ranges=True)

#FB api global vars
app_id = "1696549057338916"
app_secret = "21090405ac37194a1d4578aeb2371845" # DO NOT SHARE WITH ANYONE!
access_token = app_id + "|" + app_secret

#classifier global vars
def unpickle():
    with open('clf_driver.pkl', 'rb') as fid:
        clf_driver = pickle.load(fid)
    with open('clf_roundtrip.pkl', 'rb') as fid:
        clf_roundtrip = pickle.load(fid)
    with open('clf_relevant.pkl', 'rb') as fid:
        clf_relevant = pickle.load(fid)
    with open('vocab1.pkl', 'rb') as fid:
        vocab1 = pickle.load(fid)
    with open('vocab2.pkl', 'rb') as fid:
        vocab2 = pickle.load(fid)
    return clf_driver, clf_roundtrip, clf_relevant, vocab1, vocab2

clf_driver, clf_roundtrip, clf_relevant, vocab1, vocab2 = unpickle()

def load_data(csv_fpath):
        print("loading data")
        data = pd.read_csv(csv_fpath, encoding = 'latin1')
        return data

def request_until_succeed(url):
    req = urllib2.Request(url)
    success = False
    while success is False:
        try:
            response = urllib2.urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception, e:
            print e
            datetime.time.sleep(5)

            print "Error for URL %s: %s" % (url, datetime.datetime.now())
            print "Retrying."

    return response.read()

# Needed for tricky unicode processing
def unicode_normalize(text):
    return text.translate({ 0x2018:0x27, 0x2019:0x27, 0x201C:0x22, 0x201D:0x22,
                            0xa0:0x20 }).encode('utf-8')

def process_fb_group(group_id, home, state, k):
    set_home_location(home, state)
    #setup json dictionary
    results = {}
    dates = {}
    locations = {}
    posts = {}
    # 2nd level
    dates_drivers = defaultdict(list)
    #will create a new list object, if the key is not found in the dictionary
    dates_riders = defaultdict(list)
    locations_drivers = defaultdict(list)
    locations_riders = defaultdict(list)
    posts_drivers = {}
    posts_riders = {}

    has_next_page = True
    num_processed = 0

    statuses = getFacebookPageFeedData(group_id, access_token, 100)
    while has_next_page:
        for status in statuses['data']:
            if num_processed >= k:
                break
            # Ensure it is a status with the expected metadata
            if 'message' in status:
                status_id, status_permalinkurl, status_published,\
                status_message, status_location, status_price,\
                status_author, status_type, status_link,\
                link_name = processFacebookPageFeedStatus(status,access_token)
                if debug:
                    print(status_message)
                is_driver, is_roundtrip, is_relevant = predict(status_message)
                if is_relevant: #not spam so continue processing
                    coordinates = process_sentence(status_message, state)
                    date_list = get_dates(status_message, status_published)
                    #print(coordinates)
                    #print(date_list)
                    routes = []
                    if len(coordinates) == 1:
                        routes.append({'start':coordinates[0],
                                      'end':coordinates[0]})
                    elif len(coordinates) == 2:
                        routes.append({'start':coordinates[0],
                                      'end':coordinates[1]})
                        if len(coordinates) == 4:
                            routes.append({'start':coordinates[2],
                                      'end':coordinates[3]})
                    #print(routes)
                    if is_driver: #add to drivers
                        for date in date_list:
                            dates_drivers[date].append(status_permalinkurl)
                        for coord in coordinates:
                            latlng = (coord['lat'], coord['lng'])
                            locations_drivers[latlng].append(status_permalinkurl)
                        posts_drivers[status_permalinkurl] = {'routes': routes}
                    else: #add to riders
                        for date in date_list:
                            dates_riders[date].append(status_permalinkurl)
                        for coord in coordinates:
                            latlng = (coord['lat'], coord['lng'])
                            locations_riders[latlng].append(status_permalinkurl)
                        posts_riders[status_permalinkurl] = {'routes': routes}

            num_processed += 1

        # if there is no next page, we're done.
        if 'paging' in statuses.keys() and num_processed < k:
            statuses = json.loads(request_until_succeed(\
                    statuses['paging']['next']))
        else:
            has_next_page = False

    # 2nd level
    dates['drivers'] = dict(dates_drivers)
    dates['riders'] = dict(dates_riders)
    locations['drivers'] = remap_keys(locations_drivers)
    locations['riders'] = remap_keys(locations_riders)
    posts['drivers'] = posts_drivers
    posts['riders'] = posts_riders

    #1st level
    results['dates'] = dates
    results['locations'] = locations
    results['posts'] = posts

    #print(results)
    return results

def remap_keys(mapping):
    return [{'key':k, 'value': v} for k, v in mapping.iteritems()]

def processFacebookPageFeedStatus(status, access_token):

    # The status is now a Python dictionary, so for top-level items,
    # we can simply call the key.

    # Additionally, some items may not always exist,
    # so must check for existence first

    if 'message' in status.keys():
        #print unicode_normalize(status['message'])
        #split on \n and remove empty lines
        lines = filter(None, [s.strip() for s in status['message'].splitlines()])
        #handle Sale Posting which contain either a price and location, or just a price
        #It is a sale posting if the 2nd line of the message starts with a price or 'FREE'
        #Split the line to handle the case where someone started a line in their message
        # with a price, but it is not actually a sale post
        if len(lines) > 1 and (re.match('(^\$\d+)',
                                        [s.strip() for s in lines[1].split('-')][0])
                               or [s.strip() for s in lines[1].split('-')][0] == 'FREE'):
            #split on '-' and strip whitespace to separate out location, if there is one
            line_split = [s.strip() for s in lines[1].split('-')]
            status_price = line_split[0]
            status_location = '' if len(line_split) < 2 \
                else line_split[1]
            # set message equal to title
            status_message =  lines[0]
            # if title did not end in punctuation, append a period
            if lines[0][-1] not in '!?.,':
                status_message += '.'
            # append remaining description in sale post if they exists
            if len(lines) > 2:
                status_message += ' ' + ' '.join(lines[2:])
        else: #not a sale post
            status_location = ''
            status_price = ''
            status_message = ' '.join(lines)
        status_message = unicode_normalize(status_message)
    else:
        status_message = ''
        status_location = ''
        status_price = ''

    status_id = '' if 'id' not in status.keys() else \
        status['id']
    link_name = '' if 'name' not in status.keys() else \
            unicode_normalize(status['name'])
    status_type = '' if 'type' not in status.keys() else \
            status['type']
    status_link = '' if 'link' not in status.keys() else \
            unicode_normalize(status['link'])
    status_author = '' if 'from' not in status.keys() else \
        unicode_normalize(status['from']['name'])
    status_permalinkurl = '' if 'permalink_url' not in status.keys() else\
        status['permalink_url']

    # Time needs special care since a) it's in UTC and
    # b) it's not easy to use in statistical programs.
    if 'created_time' in status.keys():
        status_published = datetime.datetime.strptime(\
                status['created_time'],'%Y-%m-%dT%H:%M:%S+0000')
        status_published = status_published + datetime.timedelta(hours=-5) # EST
        # best time format for spreadsheet programs:
        status_published = status_published.strftime('%m/%d/%Y %H:%M')
    else:
        status_published = ''


    # return a tuple of all processed data

    return (status_id, status_permalinkurl, status_published,
            status_message, status_location, status_price,
            status_author, status_type, status_link,
            link_name)


def set_home_location(home, state):
    global home_location
    coordinates = get_coords([home_location], state)
    if coordinates:
        home_location = coordinates[0]
    else:
        home_location = ''

def process_sentence(message, state):
    tokenized = nltk.word_tokenize(message)
    tagged_sent = st.tag(tokenized)
    locations = extract_chunks(tagged_sent)
    coordinates = get_coords(locations, state)
    if len(coordinates) == 1:
        #add home location to location list
        if 'lat' in home_location and 'lng' in home_location:
            coordinates.append(home_location)
    elif len(coordinates) == 3:
        coordinates = coordinates[:2]
    elif len(coordinates) > 4:
        coordinates = coordinates[:4]
    return coordinates

def getFacebookPageFeedData(group_id, access_token, num_statuses):

    # Construct the URL string; see
    # http://stackoverflow.com/a/37239851 for Reactions parameters
    base = "https://graph.facebook.com/v2.8"
    node = "/%s/feed" % group_id
    fields = "/?fields=message,link,created_time,type,name,id,permalink_url,attachments," + \
            "comments.limit(0).summary(true),shares,reactions." + \
            "limit(0).summary(true),from"
    parameters = "&limit=%s&access_token=%s" % (num_statuses, access_token)
    url = base + node + fields + parameters

    # retrieve data
    data = json.loads(request_until_succeed(url))

    return data

def process_all(messages, state):
    for (message, _) in messages:
        process_sentence(message, state)

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
    global place_to_coords
    coordinates = []
    for location in locations:
        #construct uri
        if state != '':
            location += " " + state
        location = location.lower()
        if location not in ['back', 'campus']:
            location = location.replace('757', 'virginia beach')
            if location in place_to_coords:
                coord = place_to_coords[location]
                coordinates.append(coord)
            else:
                parameters = {'query': location, 'key': api_key}
                r = requests.get(url_base, params=parameters)
                if r.status_code == 200:
                    response = r.json()
                    if 'results' in response:
                        results = response['results']
                        #get coordinates
                        if debug:
                            print results[0]
                        if results and 'geometry' in results[0]:
                            geometry = results[0]['geometry']
                            if 'location' in geometry:
                                coord = geometry['location']
                                coordinates.append(coord)
                                place_to_coords[location] = coord

    #print(coordinates)
    return coordinates

def get_dates(message, date):
    dates = []
    date_posted = datetime.datetime.strptime(date, '%m/%d/%Y %H:%M')
    try:
        sutime.parsedate(message, date)
        times = sutime.parsedate(message, date)
        for time in times:
            val = time['value']
            if val[-2:] == 'WE':
                #friday, sat, sunday
                candidate_dates = [datetime.datetime.strptime(val[:8] + "-5", "%Y-W%W-%w"),
                                   datetime.datetime.strptime(val[:8] + "-6", "%Y-W%W-%w"),
                                   datetime.datetime.strptime(val[:8] + "-0", "%Y-W%W-%w")]
                for candidate in candidate_dates:
                    if candidate >= date_posted:
                        dates.append(candidate)
            else:
                candidate = datetime.datetime.strptime(val[:10], "%Y-%m-%d")
                if candidate >= date_posted:
                        dates.append(candidate)

    except Exception, e:
        print str(e)
    if not dates:
            dates.append(date_posted)
    dates_formatted = [date_obj.strftime('%m/%d/%Y') for date_obj in dates]
    return dates_formatted


def feature_vector(msg, vocab):
    stemmer = nltk.stem.porter.PorterStemmer()
    msg = msg.lower()
    tk_text = nltk.word_tokenize(msg)
    stemmed = [stemmer.stem(token) for token in tk_text]
    vec = np.array([[int(voc in stemmed) for voc in vocab]])
    return vec

def predict(msg):
    vec1 = feature_vector(msg, vocab1)
    vec2 = feature_vector(msg, vocab2)
    is_driver = clf_driver.predict(vec1)
    is_roundtrip = clf_roundtrip.predict(vec1)
    is_relevant = clf_relevant.predict(vec2)
    return is_driver, is_roundtrip, is_relevant


if __name__ == '__main__':
    #v = "shuffled_posts.csv"
    #data = load_data(csv)
    #print(data['status_message'][400])
    #process_all(zip(data['status_message'][400], data['status_published'][400]))
    #process_sentence('Looking for a ride back to Blacksburg from Fredericksburg/Stafford/Richmond/Chesterfield on Sunday!')
    #for (message, date) in zip(data['status_message'], data['status_published']):
    #   process_sentence(message)
    #get_times(message, date)
    #get_times(data['status_message'][400], data['status_published'][400])
    process_fb_group(36509972655, 'University of Virginia', 'virginia', 2)
