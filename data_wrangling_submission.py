# -*- coding: utf-8 -*-
"""
Spyder Editor
Data Wrangling Proyect
Open Street Map: Naples, Italy
"""

# Parsing the osm file
from lxml import etree

def getting_data(filename, tag):
    """
    Returns a list of dictionaries of attributes/child elements of a
    given tag/parent element by parsing iteratively on an OSM file.  
    Clear the element if accessing further child elements is no longer needed.
    Clear the parent element for computers performance’s sake. Drop empty rows
    """
    all_data = []
    data = etree.iterparse(filename, events = ('end',), tag = tag)
    for even, elem in data:
        record = {}
        for entry in elem:
            look_key = entry.attrib.keys()[0]
            key = entry.attrib[look_key]
            val = entry.attrib['v']
            record[key] = val 
        all_data.append(record)
    elem.clear()
    #TODO: This will throw an exception if parsing over a root.
    del elem.getparent()[0]
    #TODO: different behaviour in Python 2.7(list) than in Python 3(iterable object)
    new_file = filter(None, all_data)
    return new_file

dat = getting_data('naples_italy.osm', 'node')    

#selecting  element : cuisine
def select_group(data, str):
    """
    Return a list of dictionaires of attribute of choice. 
    Capture element of interest
    """
    return[d for d in data if str in d]

cuisine_data = select_group(dat, 'cuisine')
len(cuisine_data) #check
cuisine_data[:10] #check

#Checking for most common keys 
from collections import Counter
def most_common_keys(data, int):
    """
    Return a counter with most common keys of
    a given element.
    """
    keys = [k for d in data for k in d.keys()]
    print(Counter(keys).most_common(int))
    
most_common_keys(cuisine_data, 20)

#selecting final_list, making sure to append values
def select_entry(data, k):
    """
    Return a list of dictionaries of chosen ‘k’,
    assert that all values have been duly appended.
    A necessary condition for writing requested csv file. 
    """
    return[dict((k,v) for (k,v) in d.items()) for d in data] 

all_cuisine = select_entry(cuisine_data, 'cuisine')
len(all_cuisine)#check
all_cuisine[:10]#check

#Function discarded when migrating to Python3. not used
          
def correct_encoding(data, encoding_standard = 'latin_1'):
    """
    Return the output of select_entry() duly encoded 
    given the existence of foreign characters in the text.
    Not necessary in python3.
    """
    clean_data = [dict((k, v.encode('latin_1', 'replace')) for (k, v) in d.items()) for d in data]
    return clean_data

 
# Getting the fieldnames to proceed with csv file
def unique_keys(data):
    """
    Return a list of unique keys in order to determine fieldnames.
    A necessary condition for writing requested csv file.
    """
    unique_keys = list(set().union(*(d.keys()for d in data)))
    return unique_keys
fieldnames = unique_keys(all_cuisine)
print(fieldnames[:10]) #check

#writing the CSV file
import csv
def writing_toCSV(data):
    """
    Return a csv file. Us a csv.DictWriter to map iteratively dictionaries onto 
    corresponding row.
    """
    with open('output.csv', 'w') as output:
        writer = csv.DictWriter(output, fieldnames = fieldnames, extrasaction = 'ignore')
        writer.writeheader()
        for d in data:
            if any(d):
                writer.writerow(d)

writing_toCSV(all_cuisine)

#Reading, further pruning and cleaning with Pandas built-in functions.
import pandas as pd
df_cuisine = pd.read_csv('output.csv', encoding = 'latin_1')
df_cuisine.info()#check
#Selecting columns of interest as determined by counter
table_cuisine = df_cuisine.loc[:,['amenity', 'name', 'cuisine', 'opening_hours','addr:street',
                                  'addr:housenumber','addr:city', 'addr:postcode', 'phone', 
                                  'website', 'internet_access',
                                  'smoking', 'wheelchair']]
table_cuisine.info()#check

#Checking if phone numbers correspond to valid pattern
import re
pattern_phone = re.compile('^\+39081\d{7}') #23 False #2 True
phones = table_cuisine['phone']
result = phones.str.contains(pattern_phone)
phones.groupby(result).count() #23 false, 2 True

#Writing  functions aimed to standardized valid phone numbers
def format_phone(data):
    """
    Returning empty space if phone match regex pattern.
    """
    phone = str(data)
    clean_phone  = re.sub('\+39\s{1}081\s{1}\d{7}', phone.replace(' ', ''), phone)
    return clean_phone
     
def format_phone_1(data):
    """
    Returning empty space if phone match regex pattern.
    """
    phone = str(data)
    clean_phone = re.sub('^081\d{7}', '+39{}'.format(phone), phone)
    return clean_phone

def format_phone_final(data):
    """
    Returning valid phone number, 
    Parse previous one, 
    if not null return valid pattern
    """
    #TODO: possible to craft one function instead of three, but safer approach
    phone = str(data)
    m =  re.search(pattern_phone, phone)
    if m is not None:
        return m.group(0)
    
#Modifying gradually phone numbers
table_cuisine['phone_recode'] = table_cuisine.phone.apply(format_phone)
table_cuisine['phone_recode_1'] = table_cuisine.phone_recode.apply(format_phone_1)
table_cuisine['phone_recode_final'] =table_cuisine.phone_recode_1.apply(format_phone_final)
print(table_cuisine['phone_recode_final'][80:122])#check

#selecting cols of interesting, further pruning
cuisine = table_cuisine.loc[:,['amenity', 'name', 'cuisine', 'opening_hours','addr:street',
                                  'addr:housenumber','addr:city', 'addr:postcode', 'phone_recode_final', 
                                  'website', 'internet_access',
                                  'smoking', 'wheelchair']]
cuisine.info() #check

#renaming to avoid potential problems when quering the database
cuisine.rename(columns = {'phone_recode_final':'phone',
                          'addr:street':'street',
                          'addr:housenumber' :'housenumber',
                          'addr:city' : 'city',
                          'addr:postcode' :'postcode'}, inplace = True)
cuisine.info() #check

#Assesing and replacing missing values
import numpy as np    
cuisine.replace('', np.NaN)
# count the number of NaN values in each column
null_data = cuisine.isnull().sum()  
print(null_data)
df_null = pd.DataFrame(null_data)
df_null.describe()

#Writing to csv file
cuisine.to_csv('cuisine_table.csv', index = False, na_rep= 'NaN')

#Not used when migrating to Python 3
import chardet
chardet.detect('cuisine_table.csv')
chardet.detect('data_list')

#sqlaLchemy
# creating engine and establishing connection via sqlalchemy
from sqlalchemy import create_engine, MetaData
engine = create_engine('sqlite:///Udacity')
connection = engine.connect()
metadata = MetaData()

# creating the table
from sqlalchemy import Table, Column, String
naples_cuisine = Table('naples_cuisine', metadata,
                       Column('amenity', String()),
                       Column('name', String()),
                       Column('cuisine', String()),
                       Column('opening_hours', String()),
                       Column('street', String()),
                       Column('housenumber', String()),
                       Column('city', String()),
                       Column('postcode', String()),
                       Column('phone', String()),
                       Column('website', String()),
                       Column('internet_access', String()),
                       Column('smoking', String()),
                       Column('wheelchair', String()))
                       

metadata.create_all(engine)
print(repr(naples_cuisine))


#Populating the table
import csv
data_list =[]
with open('cuisine_table.csv', 'r' ) as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data = {'amenity':row[0],
        'name':row[1],
        'cuisine': row[2],
        'opening_hours': row[3],
        'street': row[4],
        'housenumber': row[5],
        'city': row[6],
        'postcode': row[7],
        'phone': row[8],
        'website': row[9],
        'internet_access': row[10],
        'smoking': row[11],
        'wheelchair':row[12]}
        data_list.append(data)

#inserting data
from sqlalchemy import insert
stmt = insert(naples_cuisine)
results = connection.execute(stmt, data_list)
print(results.rowcount) #check

#Exploring the database -  Query 1
from sqlalchemy import select, func
print(naples_cuisine.columns.keys())

# Explicitely create index to speed our queries.
from sqlalchemy import Index
Index('amenity', 'cuisine')

## the old way - Query 2
stmt = 'SELECT *FROM naples_cuisine'
results = connection.execute(stmt).fetchall()
first_row = results[0]
print(first_row)


#types of amenity- Query 3
stmt = select([naples_cuisine.columns.amenity, 
               func.count(naples_cuisine.columns.amenity)])
stmt = stmt.group_by(naples_cuisine.columns.amenity)
amenity_types = connection.execute(stmt).fetchall()
for row in amenity_types:
    print('{}:{}'.format(row[0], row[1]))
    
#visulaizing results Amenities
import matplotlib.pyplot as plt
import seaborn as sns

#convert query in a pandas data frame
df = pd.DataFrame(amenity_types)
df = df[1:]
df.rename(columns = {0: 'Amenity',
                     1: 'Quantity'}, inplace = True)
print(df)#check
#Plotting
plt.style.use('ggplot')
sns.set()
sns_plot = sns.barplot(x = df.Amenity, y = df.Quantity, hue=df.Amenity)
fig_1 = sns_plot.get_figure()
fig_1.savefig('amenity.png')


#Selecting types of Cuisine - Query 4
stmt = select([func.count(naples_cuisine.columns.cuisine.distinct())])
cuisine_count = connection.execute(stmt).scalar()
print(cuisine_count)#30

#Query 5
stmt = select([naples_cuisine.columns.cuisine, 
               func.count(naples_cuisine.columns.cuisine)])
stmt = stmt.group_by(naples_cuisine.columns.cuisine)
cuisine_types = connection.execute(stmt).fetchall()
for row in cuisine_types:
    print('{}:{}'.format(row[0], row[1]))

from tabulate import tabulate
#convert query in a pandas data frame
df_cuisine = pd.DataFrame(cuisine_types)
fig_2 = tabulate(df_cuisine, headers = ['Type of Food', '#'], tablefmt='fancy_grid')
print(fig_2)
#save table
np.save('cuisine_file', fig_2)
#load table
fig_2 =np.load('cuisine_file.npy')
print(fig_2)

#Finding a restaurant offering a specific type of fod
from sqlalchemy import and_
stmt = select([naples_cuisine.columns.name, naples_cuisine.columns.street,
              naples_cuisine.columns.housenumber]).where(
        and_(
                naples_cuisine.columns.amenity == 'restaurant',
                naples_cuisine.columns.cuisine.contains('seafood')
                 )
        )
seafood_restaurants = connection.execute(stmt).fetchall()
print(seafood_restaurants[:10])


#defining function:
def what_do_you_want_to_eat(type_of_food, integer):
    """
    Print selection in terms of type of food and number of
    required entries
    """
    stmt = select([naples_cuisine.columns.name,
                   naples_cuisine.columns.amenity,
                   naples_cuisine.columns.street,
                   naples_cuisine.columns.phone])
    stmt = stmt.where(naples_cuisine.columns.cuisine == type_of_food)
    stmt = stmt.limit(integer)
    #TODO: add conditional statement in case that selection is not available
    results = connection.execute(stmt).fetchall()
    for result in results:
        #TODO: Improve printing output, space bewtween entries.
        print('Name:{}\nAmenity:{}\nAddress:{}\nPhone:{}'.format(result[0],
              result[1], result[2] ,result[3]))

#Possible App        
what_do_you_want_to_eat('pizza', 3)
what_do_you_want_to_eat('regional',2)

#Scanvenger hunt
tree = etree.parse('map.osm')
root = etree.XML(etree.tostring(tree))
path = ('.//node[tag[@k = "amenity" and @v = "restaurant"]]')
list_clues = root.xpath(path)
print(list_clues)

for clue in list_clues:
    print (etree.tostring(clue, pretty_print = True, encoding = 'unicode'))
