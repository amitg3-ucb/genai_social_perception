import pandas as pd
import numpy as np
import altair as alt
import networkx as nx
import nx_altair as nxa
import pickle
import itertools

import gc
import time
import datetime as dt
import streamlit as st
import os

st.set_page_config(layout="wide")

st.markdown('# Social Perception Towards Generative AI')


sidebar = st.sidebar
sidebar.header('Select a Page to View')

page = sidebar.radio('',['Dashboard','About','Team','Tech Stack'],index = 0)

if page == 'Dashboard':

    panels = st.multiselect('Select View', ['Tweet Volume and Sentiment Over Time','Analysis of Hashtags and Popular Subtopics','Hashtag Communities as a Network'],default = ['Tweet Volume and Sentiment Over Time'
                                                                                                                                               ])

    if 'Tweet Volume and Sentiment Over Time' in panels:

        st.markdown('## <u>Tweet Volume and Sentiment Over Time </u>',unsafe_allow_html=True)

        # Read in Data
        df = pickle.load(open('subtasks/amit_eda/tweets_with_sentiment.pkl','rb'))
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Date'] = pd.to_datetime(df['Datetime'].apply(lambda x:x.date())).dt.date
        df['Day of Week'] = df['Datetime'].dt.weekday
        df['Hour'] = df['Datetime'].dt.hour
        df['Month'] = df['Datetime'].dt.month
        df['Year'] = df['Datetime'].dt.year
        df['MonthYear'] = df['Month'].astype(str) + '/' + df['Year'].astype(str)
        num_tweets = len(df)

        c1,c2,c3,c4,c5 = st.columns(5)
        # Date Range of Data
        start_date = c1.date_input('Start Date',min_value = df['Date'].min(), value = df['Date'].min(), max_value = df['Date'].max(), key = 'startdate')
        end_date = c2.date_input('End Date',min_value = start_date, value = df['Date'].max(), max_value = df['Date'].max(), key = 'enddate')
        df = df[(df['Date']>=start_date) & (df['Date']<=end_date)]

        # Subset Tweets to Visualize According to Sentiment
        desired_sentiment = c3.multiselect('Select Tweet Sentiment', ['Positive','Negative'], default = ['Positive','Negative'])
        df = df[df['Sentiment Score'].apply(lambda x: True if x <= 0 and 'Negative' in desired_sentiment else True if x > 0 and 'Positive' in desired_sentiment else False)]

        # Select for monthly faceted View
        monthly_facet = c4.selectbox('View Results by Monthly Facets', [True, False], index = 1)

        def panel1_view(df):

            if len(df) > 0:

                df = df.copy()

                #col1, blank_space ,col2 = st.columns([0.475,0.05,0.475])

                chart_start = df['Date'].min()
                chart_end = df['Date'].max()

                # -------------------------------------------------------------------------------------------------------Tweet Quantity Over Time
                tv = True
                if tv:
                    daily_volume = df.groupby('Date').count()['Datetime']
                    daily_volume.name = 'Daily Volume'

                    avg_tweet_rate = np.cumsum(daily_volume) / np.arange(1,len(daily_volume) + 1)
                    avg_tweet_rate.name = 'Average Tweet Rate'

                    avg_tweet_rate_7 = daily_volume.rolling(window = 7).mean()
                    avg_tweet_rate_7.name = '7 Day Rolling Tweet Rate'

                    tweet_quantity = pd.concat([daily_volume,avg_tweet_rate, avg_tweet_rate_7], axis = 1)
                    tweet_quantity.reset_index(inplace = True)

                    tweet_quantity = round(tweet_quantity, 2)

                    final_tweet_quantity = pd.DataFrame()

                    for col in ['Daily Volume','Average Tweet Rate','7 Day Rolling Tweet Rate']:

                        subset = tweet_quantity[['Date',col]].copy()
                        subset.columns = ['Date','Tweet Volume']
                        subset['Metric'] = col
                        final_tweet_quantity = pd.concat([final_tweet_quantity,subset])

                    selection = alt.selection_multi(fields=['Metric'], bind='legend')
                    selection2 = alt.selection_multi(fields=['Metric'], bind='legend')
                    opacity_value = 0.8

                    chart = alt.Chart(final_tweet_quantity,title = f'#GenerativeAI Tweet Volume Over Time ({chart_start} to {chart_end})').mark_line().encode(
                        x = alt.X('Date'),
                        y = alt.Y('Tweet Volume',title = 'Tweet Volume'),
                        color = alt.Color('Metric',scale=alt.Scale(
                            domain=['Daily Volume', '7 Day Rolling Tweet Rate','Average Tweet Rate','ChatGPT Release','GPT 4 Release','Avg. Daily Sentiment Score - Positive','Avg. Daily Sentiment Score - Negative','Average Daily Sentiment Over Time'],
                            range=['lightblue', '#57DCBE','#57ACDC','#9C27B0','#E91E63','#57ACDC','#E91E63','white'])),
                        tooltip = ['Date','Tweet Volume','Metric'],
                        opacity = alt.condition(selection, alt.value(opacity_value), alt.value(0.1))
                    ).interactive()

                    chart = chart.add_selection(selection)

                    products = pd.DataFrame({
                    'Date': ['2022-11-30', '2023-03-14'],
                    'Metric': ['ChatGPT Release','GPT 4 Release']
                    })

                    products['Date'] = pd.to_datetime(products['Date']).dt.date
                    products = products[products['Date'].apply(lambda x: True if chart_start <= x <= chart_end else False)]

                    if len(products) > 0:
                        chatGPT_release = alt.Chart(products).mark_rule().encode(
                        x='Date:T',
                        color=alt.Color('Metric', scale=alt.Scale(
                                domain=['ChatGPT Release', 'GPT 4 Release'],
                                range=['#9C27B0', '#E91E63'])),
                            opacity = alt.condition(selection2, alt.value(opacity_value), alt.value(0.1)),
                            tooltip = ['Date',alt.Tooltip('Metric', title = 'Product Release')]
                        ).interactive()

                        chatGPT_release = chatGPT_release.add_selection(selection2)

                        chart = chart + chatGPT_release

                    chart = chart.properties(height = 300, width = 500)

                    #col1.altair_chart(chart)
                    tweet_volume_chart = chart

                # ------------------------------------------------------------------------------------------------------TWEET SENTIMENT OVER TIME
                ts = True
                if ts:
                    tweet_quantity = df.groupby('Date').count()['Datetime']
                    tweet_quantity.name = 'Tweet Quantity'

                    daily_sentiment = df.groupby('Date').sum()['Sentiment Score']
                    daily_sentiment.name = 'Raw Daily Sentiment'

                    daily_sentiment_scaled = daily_sentiment/tweet_quantity
                    daily_sentiment_scaled.name = 'Daily Sentiment Score'

                    avg_sentiment = np.cumsum(daily_sentiment)/np.cumsum(tweet_quantity)
                    avg_sentiment.name = 'Average Sentiment Over Time'

                    sentiment_df = round(pd.concat([daily_sentiment_scaled, avg_sentiment], axis = 1),2)

                    final_sentiment_df = pd.DataFrame()

                    for col in ['Daily Sentiment Score','Average Sentiment Over Time']:

                        subset = sentiment_df[[col]].copy()
                        subset.columns = ['Sentiment Score']
                        subset['Score Type'] = col
                        final_sentiment_df = pd.concat([final_sentiment_df,subset])

                    final_sentiment_df.reset_index(inplace = True)
                    final_sentiment_df['Metric'] = final_sentiment_df.apply(lambda x: 'Avg. Daily Sentiment Score - Positive'
                                                                        if x['Sentiment Score'] >= 0 and x['Score Type'] == 'Daily Sentiment Score'
                                                                        else 'Avg. Daily Sentiment Score - Negative' if x['Sentiment Score'] <= 0 and 
                                                                        x['Score Type'] == 'Daily Sentiment Score' else 'Average Daily Sentiment Over Time',
                                                                        axis = 1)
                    final_sentiment_df['Tweet Volume'] = np.array(pd.concat([tweet_quantity,np.cumsum(tweet_quantity)]))
                    selection = alt.selection_multi(fields=['Metric'], bind='legend')
                    selection2 = alt.selection_multi(fields=['Metric'], bind='legend')

                    chart1 = alt.Chart(final_sentiment_df[final_sentiment_df['Score Type'] == 'Daily Sentiment Score']
                                    ,title = f'#GenerativeAI Tweet Sentiment Over Time ({chart_start} to {chart_end})').mark_bar().encode(
                        x = alt.X('Date'),
                        y = alt.Y('Sentiment Score',title = 'Sentiment Score'),
                        color = alt.Color('Metric', scale = alt.Scale(
                        domain = ['Avg. Daily Sentiment Score - Positive','Avg. Daily Sentiment Score - Negative','Average Daily Sentiment Over Time'],
                        range = ['#57ACDC','#E91E63','white'])),
                        tooltip = ['Date',alt.Tooltip('Sentiment Score',title = 'Daily Sentiment Score'), 'Tweet Volume'],
                        opacity = alt.condition(selection, alt.value(opacity_value), alt.value(0.1))
                    ).interactive()

                    chart1 = chart1.add_selection(selection)

                    chart2 = alt.Chart(final_sentiment_df[final_sentiment_df['Score Type'] == 'Average Sentiment Over Time'],
                                    title = f'#GenerativeAI Tweet Sentiment Over Time ({chart_start} to {chart_end})').mark_line(color = 'white').encode(
                        x = alt.X('Date'),
                        y = alt.Y('Sentiment Score',title = 'Sentiment Score'),
                        tooltip = ['Date',alt.Tooltip('Sentiment Score',title = 'Average Sentiment Over Time'),'Tweet Volume'],
                        opacity = alt.condition(selection, alt.value(opacity_value), alt.value(0.1))
                    ).interactive()
                    chart2 = chart2.add_selection(selection2)

                    chart = chart1 + chart2
                    chart = chart.properties(height = 300, width = 500)
                    #col2.altair_chart(chart)

                    tweet_sentiment_chart = chart

                # -------------------------------------------------------------------------------------Tweet Volume by Hour of Day and Day of Week
                hm = True
                if hm:
                    heatmap = df.groupby(['Day of Week','Hour']).count().iloc[:,[0]]
                    heatmap.columns = ['Tweet Volume']
                    heatmap.reset_index(inplace = True)
                    day_mapper = {0:'Monday', 1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
                    heatmap['day_str'] = heatmap['Day of Week'].map(day_mapper)
                    heatmap['hour_str'] = heatmap['Hour'].apply(lambda x:'12AM' if x == 0 else f'{x}AM' if x < 12 else '12PM' if x == 12 else f'{x-12}PM')

                    heatmap_chart = alt.Chart(heatmap, title = f'Tweet Volume by Day of Week and Hour of Day ({chart_start} to {chart_end})').mark_rect().encode(
                    y = alt.Y('day_str:O',sort=alt.EncodingSortField(field="Day of Week", order='ascending'), title = 'Weekday'),
                    x = alt.X('hour_str:O',sort=alt.EncodingSortField(field="Hour", order='ascending'), title = 'Hour'),
                    color = alt.Color('Tweet Volume:Q', title = 'Tweet Volume',scale=alt.Scale(
                            domain=list(heatmap['Tweet Volume'].quantile([0,0.05,0.5,0.95,1])),
                            range=['white','white','#57ACDC','#224457','#224457'],
                            #interpolate=method
                            )),
                    tooltip = [alt.Tooltip('day_str',title = 'Weekday'),alt.Tooltip('hour_str',title = 'Hour'),alt.Tooltip('Tweet Volume', title = 'Tweet Volume')]
                ).interactive().properties(
                    width=1000,
                    height=300)

                final_chart = (tweet_volume_chart | tweet_sentiment_chart) & heatmap_chart

                return final_chart

            else:
                return None

        if monthly_facet == False:
            st.markdown(f'### {len(df)} out of {num_tweets} Tweets Being Analyzed ({round(100*len(df)/num_tweets,2)}%)')
            chart = panel1_view(df)
            if chart: #Create 3 chart view if remaining data is non-empty
                st.altair_chart(chart)
            else:
                pass
        else:

            default = list(df['MonthYear'].unique())
            try:
                monthly_facet_choices = c5.multiselect('Select Months', default, default = default)
            except:
                monthly_facet_choices = c5.multiselect('Select Months', ['4/2022','5/2022','6/2022','7/2022','8/2022','9/2022','10/2022','11/2022',
                                                                        '12/2022','1/2023','2/2023','3/2023','4/2023'], default = ['4/2022'])

            #Plot each monthly facet according to selected months, but preserve time order
            for monthyear in ['4/2022','5/2022','6/2022','7/2022','8/2022','9/2022','10/2022','11/2022',
                                                                     '12/2022','1/2023','2/2023','3/2023','4/2023']:

                if monthyear in monthly_facet_choices:
                    subset = df[df['MonthYear'] == monthyear]
                    st.markdown(f'### {monthyear} -- {len(subset)} out of {num_tweets} Tweets Being Analyzed ({round(100*len(subset)/num_tweets,2)}%)')
                    chart = panel1_view(subset)
                    if chart:
                        st.altair_chart(chart)
                    else:
                        pass

    if 'Analysis of Hashtags and Popular Subtopics' in panels:

        st.markdown('## <u>Analysis of Hashtags and Popular Subtopics</u>',unsafe_allow_html=True)

        html_string = "<div class='tableauPlaceholder' id='viz1689706312178' style='position: relative'><noscript><a href='#'><img alt='Hashtag and Suptopic Analysis ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ha&#47;HashtagandSuptopicAnalysis&#47;HashtagandSuptopicAnalysis&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='HashtagandSuptopicAnalysis&#47;HashtagandSuptopicAnalysis' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ha&#47;HashtagandSuptopicAnalysis&#47;HashtagandSuptopicAnalysis&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1689706312178');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1500px';vizElement.style.height='1327px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1500px';vizElement.style.height='1327px';} else { vizElement.style.width='100%';vizElement.style.height='1427px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
        st.components.v1.html(html_string, width = 1500, height = 1300, scrolling = True)

    if 'Hashtag Communities as a Network' in panels:

        st.markdown('## <u>Hashtag Communities as a Network</u>',unsafe_allow_html=True)

        base_dir = 'subtasks/amit_eda/hashtag_community_subtopics/'
        community_chart_paths = os.listdir(base_dir)

        community_charts_all_data = []
        for fp in community_chart_paths:
            full_fp = base_dir + fp
            topic = fp[:-4]
            if topic == 'generativeai':
                topic = '#' + topic
            community_chart = pickle.load(open(full_fp,'rb'))
            community_charts_all_data.append([topic, community_chart])

        hashtag_chart = pickle.load(open('subtasks/amit_eda/hashtag_cooccurence_network.pkl','rb'))
        hashtag_chart = hashtag_chart.properties(height = 700, width = 700)

        subtopics = st.selectbox('Pick a Subtopic:', [a for a,b in community_charts_all_data], index = 0)
        detail_view = [b for a,b in community_charts_all_data if a == subtopics]

        c1, c2 = st.columns(2)
        c1.altair_chart(hashtag_chart)

        if len(detail_view) > 0:
            partial_network = detail_view[0]
            partial_network = partial_network.properties(height = 700, width = 700)
            c2.altair_chart(partial_network)

elif page == 'About':
    
    #####################################################
    
    st.markdown('## <u>Data</u>',unsafe_allow_html=True)
    
    st.write('<b>Tweets Being Analyzed:</b> https://www.kaggle.com/datasets/arinjaypathak/generative-ai-tweets',unsafe_allow_html=True)
    st.write('<b>Sentiment140 Dataset (Training + Evaluation of Sentiment Analysis Model):</b> https://www.kaggle.com/datasets/kazanova/sentiment140',unsafe_allow_html=True)
    
    #####################################################

    st.markdown('## <u>Summary</u>',unsafe_allow_html=True)
    
    st.write("Even since the launch of ChatGPT by OpenAI in November 2022, Generative AI as a buzzword and trend has taken the world by storm. With the ability to revolutionize nearly every industry, we have seen conversations rise across a variety of topics related to Generative AI's impact on healthcare, art, finance, law, ethics, and more. These conversations have permeated to social media sites such as LinkedIn and Twitter, and this has served as an invaluable data-driven resource for identifying patterns in social perception towards Generative AI, especially related to topic popularity, popularity of associated subtopics, and sentiment. \n\nThis project incorporates methods in statistics, machine learning, and data visualization to analyze nearly 60000 tweets with #GenerativeAI as an included hashtag between April 2022 and April 2023 with the goal of identifying changes in social perception towards Generative AI. This analysis is focused on the following key tasks:")
    
    st.markdown('1. Identifying patterns in tweet volume and sentiment over time')
    st.markdown('2. Identifying patterns in subtopic rates and associated sentiment over time')
    st.markdown('3. Identifying relationships between hashtags and hashtag communities (subtopics)')
    
    #####################################################
    
    st.markdown('## <u>Task 1: Identifying patterns in tweet volume and sentiment over time</u>',unsafe_allow_html = True)
    
    st.markdown('### Navigation')
    st.write('<b>Dashboard View:</b> Tweet Volume and Sentiment Over Time',unsafe_allow_html = True)
    
    st.markdown('### Visualizations')
    st.write('<b><u>#GenerativeAI Tweet Volume Over Time</u></b>',unsafe_allow_html = True)
    st.write('This visualization shows daily tweet quantity, rolling 7 day average tweet rate, and average tweet rate since the beginning of time, all over time. In addition, there are superimpositions for ChatGPT and GPT4 releases to visualize the impact of these large scale announcements on tweet volume.')
    
    st.write('<b><u>#GenerativeAI Tweet Sentiment Over Time</u></b>',unsafe_allow_html = True)
    st.write('This visualization shows daily average tweet sentiment (closer to 1 indicates positive sentiment, closer to 0 indicates neutral sentiment, and closer to -1 indicates negative sentiment), average tweet sentiment since the beginning of time, and a color code to easily view when average daily sentiment is positive vs negative.')
    
    st.write('<b><u>Tweet Volume by Day of Week and Hour of Day</u></b>',unsafe_allow_html = True)
    st.write('This visualization shows a heatmap that depicts tweet volume by day of week and hour of day in order to visualize temporal effects on tweet volume for a subset of posts.')
    
    st.markdown('### Example Insights')
    st.write('See below for a list of easily achievable insights from this panel. Keep in mind that this list is not comprehensive and additional insights can be drawn from this section, but this is rather intended to serve as a foundational bank of initial insights. \n\nBetween Date X and Date Y, view:')
    st.markdown('- How many tweets occured during this period?')
    st.markdown('- How did tweet volume change during this period?')
    st.markdown('- How does time of day or day of week influence tweet activity?')
    st.markdown('- How did average sentiment change over time?')
    st.markdown('- Were there any specific days that had a negative daily sentiment?')
    st.markdown('- Were there any specific days that anomalously high or low sentiments, and were there any publicized events that coincide and could help explain observed daily average sentiment for these periods?')
    
    st.markdown('### Other Notes')
    st.write('This panel includes the ability to condition studies to strictly positive and/or negative tweets. In addition, there is a monthly facet option which will reproduce the visualizations for every month that exists within the data following conditioning by date and tweet sentiment filters in order to aid in identifying patterns over time.')
    
    #####################################################
    
    st.markdown('## <u>Task 2: Identifying patterns in subtopic rates and associated sentiment over time</u>',unsafe_allow_html = True)
    
    st.markdown('### Navigation')
    st.write('<b>Dashboard View:</b> Analysis of Hashtags and Popular Subtopics',unsafe_allow_html = True)
    
    st.markdown('### Visualizations')
    st.write('<b><u>Top Subtopics</u></b>',unsafe_allow_html = True)
    st.write('This visualization shows the subtopic rate (% of tweets that were relevent to a given subtopic) for the top 23 subtopics within our dataset, conditioned by month with a monthly animation.')
    
    st.write('<b><u>Subtopic Rates and Sentiment Over Time</u></b>',unsafe_allow_html = True)
    st.write('This visualization shows the subtopic rate and average sentiment for each of the top 23 subtopics within our dataset, conditioned by month with a monthly animation that also preserves a trial to aid in identifying how topic specific sentiment and tweet volume has changed over time.')
    
    st.write('<b><u>Top 25 Hashtags</u></b>',unsafe_allow_html = True)
    st.write('This visualiation shows the hashtag rate (% of tweets with a given hashtag) for the top 25 hashtags in each month, with a monthly animation.')
    
    st.write('<b><u>Average Suptopic Rates (Size) and Sentiment (Color)</u></b>',unsafe_allow_html = True)
    st.write('This tree map visualization shows the subtopic rate for each of the top 23 subtopics, indicated by block size, and the average sentiment, indicated by color where increasingly blue shades equate to more positive average subtopic sentiment.')
    
    st.markdown('### Example Insights')
    st.write('See below for a list of easily achievable insights from this panel. Keep in mind that this list is not comprehensive and additional insights can be drawn from this section, but this is rather intended to serve as a foundational bank of initial insights. View:')
    st.markdown('- What proportion of tweets contained content related to a variety of domain specific subtopics at a single monthly frame?')
    st.markdown('- How did rates/sentiment across subtopics change over time? Did some topics rapidly become more or less popular? Did any stay consistently popular or obsolete? What about the variability in sentiment perception?')
    st.markdown('- What were the top 25 hashtags and associated hashtag rates in a particular month? ')
    st.markdown('- Were any hashtags consistently popular over time?')
    st.markdown('- What was the subtopic rate and average sentiment towards top domain specific subtopics across the entire dataset (April 2022 - 2023)?')
    
    st.markdown('### Other Notes')
    st.write('N/A')

    #####################################################
    
    st.markdown('## <u>Task 3: Identifying relationships between hashtags and hashtag communities (subtopics)</u>',unsafe_allow_html = True)
    
    st.markdown('### Navigation')
    st.write('<b>Dashboard View:</b> Hashtag Communities as a Network',unsafe_allow_html = True)
    
    st.markdown('### Visualizations')
    st.write('<b><u>Interaction Between Popular Hashtags -- Color-Code by Community</u></b>',unsafe_allow_html = True)
    st.write('This visualization shows popular hashtags as a network graph, where nodes are hashtags and edges are drawn between hashtags that co-occur. Edge weight is defined by co-occurence rate, indicated by the probability of hashtag X occurring given that hashtag Y is occurring. Hashtag communities (or subtopics) are emphasived via node color, and a community is defined by clusters of hashtags that have a tendency to co-occur.')
    
    st.write('<b><u>Detail View: Interaction Between Popular Hashtags, {Community}</u></b>',unsafe_allow_html = True)
    st.write('This visualization conditions the network graph above by an individual community / hashtag. This allows for easier identification of the hashtags that makeup a given subtopic in addition to how hashtags within a single community co-occur.')
    
    st.markdown('### Example Insights')
    st.write('See below for a list of easily achievable insights from this panel. Keep in mind that this list is not comprehensive and additional insights can be drawn from this section, but this is rather intended to serve as a foundational bank of initial insights. View:')
    st.markdown('- What are some examples of hashtags that make up particular subtopics, and does this intuitively make sense?')
    st.markdown('- What are some examples of hashtags that co-occur frequently but do not belong to the same subtopic?')
    st.markdown('- Are there any subtopics that also seem to co-occur?')
    
    st.markdown('### Other Notes')
    st.write('Subtopics in the "Analysis of Hashtags and Popular Subtopics" panel were identified using the results of this study that yielded 23 total hashtag communities / subtopic. A tweet was considered to have contained content for a given subtopic if it contained at least 1 hashtag from the associated community.')
    
    #####################################################
    
    st.markdown('## <u>Highlights on Technical Approach',unsafe_allow_html = True)
    
    st.markdown('### Sentiment Analysis')
    st.write('Tweet sentiment was determined using a custom trained deep learning model which was trained on a binary label version of the Sentiment140 dataset. Our binary classification model was trained on approximately 1.2M tweets and was evaluated on nearly 400K tweets. We utilized HuggingFace Pretrained SentenceTransformer to embed the tweets as model inputs rather than training our own word embeddings and having a custom network develop a contextual embedding representation of the input.\n\nOur model acheived an out of sample accuracy of nearly 83% on the 400K tweets in the hold out set, which had an approximatley 50-50 label distribution, and acheived an accuracy of slightly over 90% over a manual evaluation of nearly 1000 tweets in our #GenerativeAI dataset. A continuous sentiment score was acheived by computing (2*P(Positive Sentiment | X) - 1) = P(Positive Sentiment | X) - P(Negative Sentiment | X). This enabled scores near 1 for highly positive sentiment tweets, scores near 0 for fairly neutral tweets, and scores near -1 for highly negative sentiment tweets.')
    
    st.markdown('### Subtopic / Hashtag Community Identification')
    st.write('In order to identify hashtag communities, defined via hashtag groups that have a tendency to co-occur, we first developed a Hashtag Co-Occurence Matrix. This stored, for every i,j position, the probability of the ith hashtag occuring given that the jth hashtag had occurred. This matrix was directly used to form the network visualization in the "Hashtag Communities as a Network" panel.\n\nFollowing the development of the co-occurence matrix, we utilized a custom implementation of a DBSCAN algorithm to traverse the matrix and identify clusters have hashtags that have a tendency to co-occur. After these clusters were identified, we observed the hashtag in each to assign a logical label or subtopic. These subtopics were used as node color labels and motivated the detail view in the "Hashtag Communities as a Network" panel, and were also used to identify suptoptic rates/sentiment over time in the "Analysis of Hashtags and Popular Subtopics" panel.')

elif page == 'Tech Stack':
    
    c1, c2 = st.columns(2)
    
    c1.markdown('### Primary Programming Language(s)')
    c1.markdown('- Python')
    
    c1.markdown('### User Interface Packages and Products')
    c1.markdown('- Streamlit')
    
    c1.markdown('### Data Visualization Packages and Products')
    c1.markdown('- Altair')
    c1.markdown('- networkX')
    c1.markdown('- nx_altair')
    c1.markdown('- Tableau')
    
    c1.markdown('### Data Preprocessing')
    c1.markdown('- numpy')
    c1.markdown('- pandas')
    
    c1.markdown('### Sentiment Analysis')
    c1.markdown('- HuggingFace')
    c1.markdown('- TensorFlow/Keras')
    
    c1.markdown('### Hashtag Community / Subtopic Identification')
    c1.markdown('- numpy')
    c1.markdown('- pandas')
    
    img_size = 100
        
    c2.markdown(f'<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" jsaction="VQAsE" class="r48jcc pT0Scc iPVvYb" style="max-width: 1869px; height: {img_size}px; margin: 0px; width: {img_size}px;" alt="File:Python-logo-notext.svg - Wikimedia Commons" jsname="kn3ccd" aria-hidden="false">',unsafe_allow_html = True)
    
    c2.markdown(f'<img src="https://streamlit.io/images/brand/streamlit-logo-primary-lightmark-lighttext.png" jsaction="VQAsE" class="r48jcc pT0Scc iPVvYb" style="max-width: 2181px; height: {img_size}px; margin: 0px; width: 110px;" alt="Brand • Streamlit" jsname="kn3ccd" aria-hidden="false">',unsafe_allow_html = True)
    
    c2.markdown(f'<img src="https://altair-viz.github.io/_static/altair-logo-light.png" jsaction="VQAsE" class="r48jcc pT0Scc iPVvYb" style="max-width: 200px; height: {img_size}px; margin: 0px; width: {img_size}px;" alt="Vega-Altair: Declarative Visualization in Python — Vega-Altair 5.0.1  documentation" jsname="kn3ccd" aria-hidden="false">',unsafe_allow_html = True)
    
    c2.markdown(f'<img src="https://avatars.githubusercontent.com/u/388785?s=280&amp;v=4" jsaction="VQAsE" class="r48jcc pT0Scc iPVvYb" style="max-width: 280px; height: {img_size}px; margin: 0px; width: {img_size}px;" alt="NetworkX · GitHub" jsname="kn3ccd" aria-hidden="false">',unsafe_allow_html = True)
    
    c2.markdown(f'<img src="https://nextviewconsulting.com/sites/default/files/styles/large/public/icons/logo-tableau-cirkel.png?itok=CsAZTLUk" jsaction="VQAsE" class="r48jcc pT0Scc iPVvYb" style="max-width: 360px; height: {img_size}px; margin: 0px; width: {img_size}px;" alt="Tableau | Nextview Consulting" jsname="kn3ccd" aria-hidden="false">',unsafe_allow_html = True)
    
    c2.markdown(f'<img src="data:image/png;base64,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" jsaction="VQAsE" class="r48jcc pT0Scc" alt="NumPy" jsname="JuXqh" style="max-width: 800px; width: {img_size}px; height: {img_size}px; margin: 0px;" data-iml="11215.10000000149">',unsafe_allow_html = True)
    
    c2.markdown(f'<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAAEDCAMAAABQ/CumAAAAeFBMVEX///8TB1QAAEb/ygDnBIgPAFLNzNYTAFnQ0NgMAFcAAETb2eP39/oUBlfV1N7/xwDmAID/9tfLydcjG17/4Yz//vbCwM3ykcL61OfoBIwyKmgAADYAAE0AAErx8PTIxdT/+un/34T85/Lyir/lAHv50eX+9fkpH2Ma8J+4AAACEklEQVR4nO3dzVIaQRSAUYNCEIGoiYmJivnP+79hFrmLVHELZ6pnmG483xqaPruh5lb32ZkkSZIkSZIkvb52z7dZU2+rT4uH2X6rx6m31afF7M1+87dTb6tPCDWEUEMINYRQQ5MS1tu0nqtMSrhKn26e1v1WmZawyn58g4DQL4QIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECOFA6cvM5a4nYb29yjoO4WmVvM58WPQkbF8e+RqPcDlPVp4t+xLS/W0QEBCqI8yTLpsizN8n/WmJ0CEEBAQEBAQEBIT2CF+/fci6a4hw8y7rvC3CeRYCAgICAgICAgICAgICwlCEtJYIdzdp/3+kdkKHToFQ+RjJMCEcCKF7CAdC6B7CgRC6Nylh9zGtJUJ6uNCsnsOFhhkvPAHC9x+fsloi/Pp5nXTREuH++iLpMwICAgICAgICAgICAgKC/87R7/u0lggdQkBAQEBAQEB4dYQON67UTqh9KuwkDlRBQED4R8gOF5o3Rdh8yepLGO0ez6MNPO+WQ9w3NilhvBAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyECKEshAihLIQIoSyEKJt+lL0SNeADUR4TG9cGWXHew10AkPP4aRBO9ohEuOFUEMINYRQQwg1dAKEDvd41t5t2u7lL0qSJEmSJEnSyfUXeomSFq0EzbkAAAAASUVORK5CYII=" jsaction="VQAsE" class="r48jcc pT0Scc" alt="pandas - Python Data Analysis Library" jsname="JuXqh" style="max-width: 600px; width: {img_size}px; height: {img_size}px; margin: 0px;" data-iml="3697.6999999955297">',unsafe_allow_html = True)
    
    c2.markdown(f'<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" jsaction="VQAsE" class="r48jcc pT0Scc iPVvYb" style="max-width: 1024px; height: {img_size}px; margin: 0px; width: {img_size}px;" alt="Brand assets - Hugging Face" jsname="kn3ccd" aria-hidden="false">',unsafe_allow_html = True)
    
    c2.markdown(f'<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/1200px-TensorFlow_logo.svg.png" jsaction="VQAsE" class="r48jcc pT0Scc iPVvYb" style="max-width: 1200px; height: {img_size}px; margin: 0.5px 0px; width: {img_size}px;" alt="File:TensorFlow logo.svg - Wikimedia Commons" jsname="kn3ccd" aria-hidden="false">',unsafe_allow_html = True)

elif page == 'Team':
    
    st.write('Our team is comprised of 3 members as part of the DATASCI209 Graduate Data Visualization Course within the MIDS program at UC Berkeley (Summer Trimester, 2023):')
    
    st.markdown('- Amit Gattadahalli, LinkedIn: https://www.linkedin.com/in/amitg3/')
    st.markdown("- Amanda D'Alessio, LinkedIn: https://www.linkedin.com/in/amanda-d-alessio/")
    st.markdown("- Abdulaziz (Jai-Jai) Macabangon, LinkedIn: https://www.linkedin.com/in/abdulazizmacabangon/")


   
        
    
    