
# core packages
import streamlit as st 
import streamlit.components.v1 as components
st.set_page_config(layout="wide")

# data analysis packages
import pandas as pd
import numpy as np
import matplotlib
import itertools
from itertools import chain
from itertools import repeat

# for network 
import networkx as nx
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool
import xarray as xr
import datashader as ds
import skimage
from holoviews.operation.datashader import datashade, bundle_graph

hv.extension('bokeh')
defaults = dict(width=600, height=600)
hv.output(widget_location='top_left')
hv.opts.defaults(
    opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))


# for NLP analysis
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

# variables
font_size_labels = 30
sns_font_scale = 2


process_list=['Formulation','Synthesis','Analysis','Evaluation',
             'Documentation','Reformulation 1','Reformulation 2','Reformulation 3']

process_list_graph=['Synthesis','Analysis','Evaluation','Reformulation 1','Reformulation 2']

interaction_list = ['A>A','A>B','B>A','B>B','B>C','C>B','C>C','C>A','A>C']

#some common non-stopwords that are irrelevant to engineering design
irrelevant_words = ["right","us","thought","said","think","say","thank","went","yeah","yes","kind",
    				"awesome","excellent","great","like","ok","okay","so","cool","sure","thing",
 					"go","went","get","also","got","would","could","tri","guy","dude", "soon","stuff",
   					 "huh","unintelligible",'hey',"want","know","mean","maybe","yep","mmhmm"]

#dictionary of contractions or slangs to convert back to original form
contractions_slangs = {"aren't": "are not","can't": "can not","can't've": "can not have","cannot": "can not",
						"'cause": "because","could've": "could have","couldn't": "could not",
						"couldn't've": "could not have","didn't": "did not","doesn't": "does not",
						"don't": "do not","hadn't": "had not","hadn't've": "had not have",
						"hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have",
						"he'll": "he will","he'll've": "he will have","he's": "he is","how'd": "how did",
						"how'd'y": "how do you","how'll": "how will","how's": "how is","i'd": "i would",
						"i'd've": "i would have","i'll": "i will","i'll've": "i will have",	"i'm": "i am",
						"i've": "i have","isn't": "is not","it'd": "it would","it'd've": "it would have",
						"it'll": "it will","it'll've": "it will have","it's": "it is","let's": "let us",
						"ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
						"mightn't've": "might not have","must've": "must have","mustn't": "must not",
						"mustn't've": "must not have","needn't": "need not","needn't've": "need not have",
						"o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have",
						"shan't": "shall not","shan't've": "shall not have","she'd": "she would",
						"she'd've": "she would have","she'll": "she will","she'll've": "she will have",
						"she's": "she is","should've": "should have","shouldn't": "should not",
						"shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would",
						"that'd've": "that would have","that's": "that is","there'd": "there would",
						"there'd've": "there would have","there's": "there is","they'd": "they would",
						"they'd've": "they would have","they'll": "they will","they'll've": "they will have",
						"they're": "they are","they've": "they have","to've": "to have","wasn't": "was not",
						"we'd": "we would","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
						"we're": "we are","we've": "we have","weren't": "were not","what'll": "what will",
						"what'll've": "what will have","what're": "what are","what's": "what is","what've": "what have",
						"when's": "when is","when've": "when have","where'd": "where did","where's": "where is",
						"where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is",
						"who've": "who have","why's": "why is","why've": "why have","will've": "will have",
						"won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
						"wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would",
						"y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
						"you'd": "you would","you'd've": "you would have","you'll": "you will",
						"you'll've": "you will have","you're": "you are","you've": "you have","gonna": "going to",
						"wanna": "want to"}


# ////////////////////////////////  FUNCTION NLP  //////////////////////////////////////

#fonction to create tokens
def create_token(text):
    # convert all letters to lower case
    try:
        text = text.lower()
    except AttributeError:
        print("Text causing error: ", text)
    # convert contraction
    for word in contractions_slangs:
        if word in text:
            text = text.replace(word, contractions_slangs[word])
    stopwords = nltk.corpus.stopwords.words('english')
    text_new = "".join([i for i in text if i not in string.punctuation])
    tokenized_text = nltk.tokenize.word_tokenize(text_new)

    #get rid of numbers and empty strings
    tokenized_text = [word for word in tokenized_text if (bool(re.match("[0-9]+", word)) == False) 
                      and (word != "...") and (word != "")]
    
    #get rid of stopwords
    tokenized_text = [word for word in tokenized_text if (word not in stopwords) 
                      and (word not in irrelevant_words)]
    #get rid of one letter word
    tokenized_text = [word for word in tokenized_text if len(word) > 2]
    
    tagged = nltk.pos_tag(tokenized_text)
    token_select = [token for (token,pos_tag) in tagged if pos_tag == 'NN' or pos_tag == 'NNS']
    
    #stemming tokens
    ps = PorterStemmer()
    tokens_stem=[]
    for word in token_select: 
        temp = ps.stem(word)
        tokens_stem.append(temp)
    
    return(tokens_stem)

def return_first_occurance (data):
    list_tokens = []
    for lines in range (len(data)):
        segment_token = data.Token[lines]
        list_previous_tokens = list(itertools.chain(*data.Token[:lines].values))
    
        set_tokens = set(list_previous_tokens)
        temp = [x for x in segment_token if x not in set_tokens]
        list_tokens.append(temp)
    return(list_tokens)

def get_unique_token(token_list):

    list_unique = []
    unique_token = set(token_list)
    for token in unique_token:
        list_unique.append(token)

    return(list_unique)

# fonction to add a column to files with the token created
def add_token_column(data):
    token_list=[]
    for i in range (len(data)):
        temp=create_token(data['Utterance'][i])
        token_list.append(temp)
    return(token_list)

# ////////////////////////////////  FUNCTION CREATE GRAPH  //////////////////////////////////////

def create_graph_FO(data):
    
    temp = add_token_column(data)
    data['Token'] = temp
    
    # create column with FO value
    fo_token = return_first_occurance(data)
    
    list_fo_unique = []
    for j in range (len(fo_token)):
        fo_unique = get_unique_token(fo_token[j])
        list_fo_unique.append(fo_unique)
    
    data['Token_FO'] = list_fo_unique
    
    # drop token
    count_token = []
    for rows in range(len(data)):
        temp = len(data.Token_FO[rows])
        count_token.append(temp)
    data['count_token'] = count_token
    data_drop = data[data.count_token > 0]
    data_drop = data_drop.reset_index()
    
    # create node source
    source_list=[]
    for rows in range (len(data_drop)-1):
        temp = data_drop.Token_FO[rows] * len(data_drop.Token_FO[rows + 1])
        source_list.append(temp)
        source_list_flat = list(itertools.chain(*source_list))
    
    # create node target
    target_list=[]
    for rows in range (1, len(data_drop), 1):
        for token in range (len(data_drop.Token_FO[rows])):
            temp = [data_drop.Token_FO[rows][token]]
            temp0 = [x for item in temp for x in repeat(item, len(data_drop.Token_FO[rows-1]))]
            target_list.append(temp0)
            target_list_flat = list(itertools.chain(*target_list))
    
    # create time 
    time_normalize = np.linspace(1,3, num=len(source_list_flat)).astype(int)
    
    # create process information for edges
    source_process=[]
    for rows in range (len(data_drop)-1):
        for token in range (len(data_drop.Token_FO[rows])):
            temp = [data_drop['Code FBS'][rows]]
            temp0 = [x for item in temp for x in repeat(item, len(data_drop.Token_FO[rows+1]))]
            source_process.append(temp0)
            source_process_flat = list(itertools.chain(*source_process))
            
    target_process = []
    for rows in range (1, len(data_drop), 1):
        for token in range (len(data_drop.Token_FO[rows])):
            temp = [data_drop['Code FBS'][rows]]
            temp0 = [x for item in temp for x in repeat(item, len(data_drop.Token_FO[rows-1]))]
            target_process.append(temp0)
            target_process_flat = list(itertools.chain(*target_process))
            
    process=[]
    for i in range(len(source_process_flat)):
        temp=[str(source_process_flat[i])+str(target_process_flat[i])]
        process.append(temp)

    df_process= pd.DataFrame(process, columns = ['Process'])
   
    # rename FBS processes
    df_process['Process'] = df_process['Process'].replace(['RF','FBe'],'Formulation')
    df_process['Process'] = df_process['Process'].replace(['BeS'],'Synthesis')
    df_process['Process'] = df_process['Process'].replace(['SBs'],'Analysis')
    df_process['Process'] = df_process['Process'].replace(['BsBe'],'Evaluation')
    df_process['Process'] = df_process['Process'].replace(['BeBs'],'Evaluation')
    df_process['Process'] = df_process['Process'].replace(['SD'],'Documentation')
    df_process['Process'] = df_process['Process'].replace(['SS'],'Reformulation 1')
    df_process['Process'] = df_process['Process'].replace(['SBe'],'Reformulation 2')
    df_process['Process'] = df_process['Process'].replace(['SF'],'Reformulation 3')
    
    # create speaker infor for edges
    source_speaker=[]
    for rows in range (len(data_drop)-1):
        for token in range (len(data_drop.Token_FO[rows])):
            temp = [data_drop['Speaker'][rows]]
            temp0 = [x for item in temp for x in repeat(item, len(data_drop.Token_FO[rows+1]))]
            source_speaker.append(temp0)
            source_speaker_flat = list(itertools.chain(*source_speaker))

    target_speaker = []
    for rows in range (1, len(data_drop), 1):
        for token in range (len(data_drop.Token_FO[rows])):
            temp = [data_drop['Speaker'][rows]]
            temp0 = [x for item in temp for x in repeat(item, len(data_drop.Token_FO[rows-1]))]
            target_speaker.append(temp0)
            target_speaker_flat = list(itertools.chain(*target_speaker))
     
    speaker=[]
    for i in range(len(source_speaker_flat)):
        temp=[str(source_speaker_flat[i])+'>'+str(target_speaker_flat[i])]
        speaker.append(temp)

    df_speaker= pd.DataFrame(speaker, columns = ['Speaker'])   
    
    #create graph from networkx
    df_edges = pd.DataFrame({"source": source_list_flat, "target": target_list_flat, 
                         "time":time_normalize, "process": df_process.Process.tolist(),
                         "interaction": df_speaker.Speaker.tolist()})
    G = nx.from_pandas_edgelist(df_edges, edge_attr=True)
    graph_temp = hv.Graph.from_networkx(G, nx.layout.fruchterman_reingold_layout, k = 0.7)

    # create list of node size
    count_node = df_edges.source.value_counts()
    df_count = pd.DataFrame(count_node)
    df_count_node = df_count.reset_index()
    df_count_node.columns = ['node', 'occurence']

    # recreate graph for better display / handling
    x, y, node = graph_temp.nodes.array([0, 1, 2]).T
    df_node = pd.DataFrame({'x': x ,'y':y ,'node': node })

    df_node_new = df_node.merge(df_count_node, on = 'node', how ='outer')
    df_node_new_fill = df_node_new.fillna(1)
    #df_node_new_fill['label'] = df_node_new_fill['node']+ '-' + df_node_new_fill['occurence'].map(str)

    g_nodes = hv.Nodes(df_node_new_fill).sort()

    graph = hv.Graph((df_edges, g_nodes))

    graph.opts(tools=['hover','wheel_zoom','reset'], active_tools=['wheel_zoom'], 
           node_size=4, cmap = 'blues', node_color='white', node_line_color='lightgrey',
           edge_color='lightgrey', edge_line_width = 0.5, xaxis=None, yaxis=None)
           #directed=True, arrowhead_length=0.01)

    # bundle edges
    bundled = bundle_graph(graph)
    return(bundled, graph)



# ////////////////////////////////  FILE UPLOAD NLP  //////////////////////////////////////

session_01 = pd.read_csv("https://raw.githubusercontent.com/Julie-Milovanovic/design-space-map/main/assets/session01.csv")
#session_02 = pd.read_csv("https://raw.githubusercontent.com/Julie-Milovanovic/design-space-map/main/assets/session02.csv")



# ////////////////////////////////  DISPLAY ON PAGE  //////////////////////////////////////

# Organization page
st.title("Representation of the design space for one design session of a team")
st.markdown("Select process on the side panel to characterize the design space.")
st.subheader("About the graph")
st.markdown("This graph represents new concepts generated by designers overtime. Connections represent design processes that connect two ideas. The design processes are generated using the Function Behavior Structure ontology.")


col1, col2 = st.columns(2)

# //////////////////////////////// NLP / GRAPH ANALYSIS  //////////////////////////////////////


# Organization sidebar
st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use this panel to select a process")
st.sidebar.subheader("Select items")


process_list = ['Synthesis','Analysis','Evaluation','Reformulation 1','Reformulation 2']

#session_select = st.sidebar.selectbox('Session', session_list)
process_selected = st.sidebar.selectbox('Process', process_list)

with col1:
    st.header("Time 1")
    graph_bundled = create_graph_FO(session_01)

    graph_select = graph_bundled[0].select(process=process_selected, time = 1)
    graph_select.opts(node_size=8,node_line_color='goldenrod',
                 edge_color='goldenrod', edge_line_width = 1, xaxis=None, yaxis=None)

    labels = hv.Labels(graph_select.nodes, ['x', 'y'], 'node')
    labels.opts(text_font_size='8pt', text_color='black', bgcolor='white',yoffset=0.015,text_align='center ')

    graph_bokeh = graph_bundled[0]*graph_select*labels

    renderer=hv.renderer('bokeh')
    html=renderer.static_html(graph_bokeh)    
    components.html(html, height=700)

    graph_select_count = graph_bundled[1].select(process=process_selected, time = 1)
    st.markdown('The number of node is '+ str(len(graph_select_count.nodes.array()))+'.')
    st.markdown('The number of edge is '+ str(len(graph_select_count.edgepaths.array()))+'.')

with col2:
    st.header("Time 2")
    graph_select = graph_bundled[0].select(process=process_selected, time = 2)
    graph_select.opts(node_size=8,node_line_color='darkseagreen',
                 edge_color='darkseagreen', edge_line_width = 1, xaxis=None, yaxis=None)

    labels = hv.Labels(graph_select.nodes, ['x', 'y'], 'node')
    labels.opts(text_font_size='8pt', text_color='black', bgcolor='white',yoffset=0.015,text_align='center ')

    graph_bokeh = graph_bundled[0]*graph_select*labels

    renderer=hv.renderer('bokeh')
    html=renderer.static_html(graph_bokeh)    
    components.html(html, height=700)
    
    graph_select_count = graph_bundled[1].select(process=process_selected, time = 2)
    st.markdown('The number of node is '+ str(len(graph_select_count.nodes.array()))+'.')
    st.markdown('The number of edge is '+ str(len(graph_select_count.edgepaths.array()))+'.')

st.caption("This material is based upon work supported by the National Science Foundation under Grant No. CMMI-1762415. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation")

