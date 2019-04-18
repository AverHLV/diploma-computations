# coding: utf8
import pandas as pd
from lxml.etree import fromstring, HTMLParser
from re import sub, search
from json import loads
from unipath import Path
from lda import LDA

ENCODING = 'utf8'
UKR_ALPHABET = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя '


def parse_html(path, lda_model, first_of=None, filename=None):
    """
    Parse vote results in html format

    :param path: unipath.Path object with data folder absolute path
    :param lda_model: .lda.LDA object
    :param first_of: iterate over %first_of% first files in data folder
    :param filename: csv filename for saving created pd.DataFrame
    """

    assert isinstance(path, Path), 'path should be an instance of unipath.Path class'
    assert isinstance(lda_model, LDA), 'lda_model should be an instance of lda.LDA class'

    # load deputies

    with open(path.listdir()[0], encoding=ENCODING) as html_file:
        parsed_df = pd.read_html(html_file)[0]

    deputies = (parsed_df[0].append(parsed_df[2], ignore_index=True)).sort_values()

    # rename duplicates

    duplicates = deputies.duplicated()

    for index in duplicates[duplicates].index:
        deputies.loc[index] = deputies.loc[index] + '_' + str(index)

    # load factions dictionary and create descriptors list

    factions = load_factions()
    unique_descriptors = ['Time', 'Topic']
    faction_descriptors = list(factions.keys())

    # create empty pd.DataFrame with columns by descriptors list

    df = pd.DataFrame(columns=unique_descriptors + faction_descriptors + deputies.tolist())

    # populate previously created DataFrame

    if first_of is None:
        files = path.listdir()
    else:
        files = path.listdir()[:first_of]

    for file in files:
        if not file.isfile():
            continue

        print('Processing {0}'.format(file))

        with open(file, encoding=ENCODING) as html_file:
            tree = fromstring(html_file.read(), HTMLParser())
            vote_time = parse_time_from_html(tree)
            vote_topic = parse_topic_from_html(tree)
            parsed_df = pd.read_html(html_file)[0]

        vote_results = pd.DataFrame()
        vote_results['Deputies'] = parsed_df[0].append(parsed_df[2], ignore_index=True)
        vote_results['Results'] = parsed_df[1].append(parsed_df[3], ignore_index=True)
        vote_results = vote_results.sort_values('Deputies')

        # get vote_results representation as list of [deputy, vote_result]

        vote_results = vote_results.to_dict(orient='split')['data']

        # parse vote results

        row = [vote_time, lda_model.predict([vote_topic])[0][0]] + [0] * len(faction_descriptors)

        for result in vote_results:
            if pd.isnull(result[1]):
                vote_results.remove(result)
                continue

            result[1] = sub(r'[^{0}]'.format(UKR_ALPHABET), '', result[1].lower())

            if result[1] == 'за':
                result[1] = 1

            elif result[1] == 'проти':
                result[1] = 0

            elif result[1] == 'утримався' or result[1] == 'утрималась':
                result[1] = 2

            elif result[1] == 'не голосував' or result[1] == 'не голосувала':
                result[1] = 3

            elif result[1] == 'відсутній' or result[1] == 'відсутня':
                result[1] = 4

            else:
                vote_results.remove(result)

            # count the number of deputies present by factions

            if result[1] != 4:
                for faction in factions:
                    if result[0] in factions[faction]:
                        row[len(unique_descriptors) + faction_descriptors.index(faction)] += 1
                        break

        vote_results = dict(vote_results)

        # insert values by list

        for column in df.columns[len(unique_descriptors) + len(faction_descriptors):]:
            try:
                row.append(vote_results[column])

            except KeyError:
                row.append(None)

        df.loc[len(df)] = row

    # fill None with most frequent values

    columns_to_drop = []

    for column in df.columns[len(unique_descriptors) + len(faction_descriptors):]:
        mode_value = df[column].mode().max()

        if pd.isnull(mode_value):
            columns_to_drop.append(column)
            continue

        df[column] = df[column].fillna(value=mode_value)

    # drop columns with only None values

    if len(columns_to_drop):
        df = df.drop(columns_to_drop, axis=1)

    # make necessary casts

    try:
        for column in df.columns[len(unique_descriptors):]:
            df[column] = df[column].astype('int')

        df['Topic'] = df['Topic'].astype('category').cat.rename_categories(range(len(df['Topic'].unique())))
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y')

    except ValueError:
        if filename is not None:
            df.to_csv(filename, index=False)

        return

    # sort rows by 'Time' column

    df = df.sort_values('Time')

    # save DataFrame

    if filename is not None:
        df.to_csv(filename, index=False)

    print('\n', df)


def parse_vote_topics(path, first_of=None, filename=None):
    """
    Parse vote topics from html files to pd.DataFrame and save it

    :param path: unipath.Path object with data folder absolute path
    :param first_of: iterate over %first_of% first files in data folder
    :param filename: csv filename for saving created pd.DataFrame
    """

    topics = []
    df = pd.DataFrame(columns=['topic'])

    if first_of is None:
        files = path.listdir()
    else:
        files = path.listdir()[:first_of]

    for file in files:
        if not file.isfile():
            continue

        print('Processing {0}'.format(file))

        with open(file, encoding=ENCODING) as html_file:
            tree = fromstring(html_file.read(), HTMLParser())
            topics.append(parse_topic_from_html(tree))

    df['topic'] = list(set(topics))

    if filename is not None:
        df.to_csv(filename, index=False)

    print('\n', df)


def parse_time_from_html(tree):
    """ Parse vote time from html """

    vote_time = search(r'\d{2}\.\d{2}\.\d{4}', tree.xpath('//span[@class="rvts2"]/text()')[0])
    return vote_time.group()


def parse_topic_from_html(tree):
    """ Parse and clear vote topic from html """

    vote_topic = tree.xpath('//span[@class="rvts1"]/text()')[1]
    vote_topic = sub(r'[^{0}]'.format(UKR_ALPHABET), '', vote_topic.lower())
    return sub(r' {2,}', ' ', vote_topic)


def load_factions(filename='json/factions.json'):
    """ Load factions json as dictionary """

    with open(filename, encoding=ENCODING) as file:
        factions = loads(file.read())

    for key in factions:
        if factions[key][0][0] == '\ufeff':
            factions[key][0] = factions[key][0][1:]

    return factions


if __name__ == '__main__':
    data_path = Path(__file__).absolute().ancestor(2).child('Data').child('html')

    # parse_vote_topics(data_path, first_of=100)
    parse_html(data_path, LDA(), first_of=10)
