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

st.set_page_config(layout="wide")

st.markdown('# Social Perception Towards Generative AI')

panels = st.multiselect('Select View', ['Tweet Quantity and Sentiment Over Time','Analysis of Hashtags and Popular Subtopics','Hashtag Communities as a Network'],default = ['Tweet Quantity and Sentiment Over Time'
                                                                                                                                           ])

if 'Tweet Quantity and Sentiment Over Time' in panels:
    
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

                chart = chart.properties(height = 300, width = 650)

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
                chart = chart.properties(height = 300, width = 650)
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
                width=1500,
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

    hashtag_chart = pickle.load(open('subtasks/amit_eda/hashtag_cooccurence_network.pkl','rb'))
    clustered_results_final = pickle.load(open('subtasks/amit_eda/interpretable_hashtag_labels.pkl','rb'))
    network_weights = pickle.load(open('subtasks/amit_eda/final_hashtag_adjacency_weights.pkl','rb'))

    subtopics = st.multiselect('Select Subtopics', list(clustered_results_final['Label'].unique()), default = ['#generativeai','Biotech'])
    clustered_results_final = clustered_results_final[clustered_results_final['Label'].isin(subtopics)]
    final_color_codes = pickle.load(open('subtasks/amit_eda/network_colors.pkl','rb'))

    hash_indices = list(clustered_results_final.index)
    network_weights = network_weights.loc[hash_indices,hash_indices]

    def create_network_graph(weights):

        n,p = weights.shape

        if n > 1 and p > 1:

            #Create Network Graph
            G = nx.from_pandas_adjacency(weights)
            # Compute positions for viz.
            pos = nx.spring_layout(G)
            #centrality helper
            helper = weights.mean(axis = 1)
            helper_array = np.array(helper)

            for n in G.nodes():
                G.nodes[n]['Hashtag'] = n
                G.nodes[n]['Community'] = clustered_results_final.loc[n,'Label']
                G.nodes[n]['Relative Centrality'] = round(100*(helper_array <= helper[n]).mean(),2)
                G.nodes[n]['Color'] = final_color_codes[clustered_results_final.loc[n,'Label']]

            alt.data_transformers.disable_max_rows()

            viz = nxa.draw_networkx(G,
                                    pos=pos,
                                    node_tooltip=['Hashtag', 'Community','Relative Centrality'],
                                    node_color = 'Color',
                                    node_label = 'Hashtag',
                                    width = 'weight',
                                    
                                    font_size = 8,
                                    edge_color = 'white',
                                    font_color = 'white'
                                )

            # Show it as an interactive plot!
            viz = viz.interactive().properties(height = 800, width = 800,title = {'text':['Detail View: Interaction Between Popular Hashtags -- Color-Coded by Hashtag Community']})

            viz = viz.configure_title(
                #fontSize=15,
                #font='Courier',
                anchor='start',
                color='white'
            )

            selection = alt.selection_single(on='mouseover', fields=['Hashtag'], empty='none')

            viz = viz.encode(
                opacity = alt.condition(selection, alt.value(0.5), alt.value(1)),
            ).add_selection(selection)

            viz = viz.configure_legend(disable = True)
            return viz
        else:
            return None

    detail_network_chart = create_network_graph(network_weights)

    c1, c2 = st.columns(2)

    c1.altair_chart(hashtag_chart)
    if detail_network_chart:
        c2.altair_chart(detail_network_chart)
        

        
    
    