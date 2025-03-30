import base64
import streamlit as st
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("background.jpg")
# data:image/png;base64,{img}
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
width: 100%;
height:100%
background-repeat: no-repeat;
background-attachment: fixed;
background-size: cover;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
teams =['--- select ---','Delhi Capitals', 'Rajasthan Royals', 'Chennai Super Kings',
       'Mumbai Indians', 'Sunrisers Hyderabad', 'Kolkata Knight Riders',
       'Royal Challengers Bangalore', 'Kings XI Punjab']

cities =['Mumbai', 'Kolkata', 'Ahmedabad', 'Hyderabad', 'Delhi',
       'Visakhapatnam', 'Centurion', 'Port Elizabeth', 'Bangalore',
       'Dharamsala', 'Jaipur', 'Raipur', 'Sharjah', 'Nagpur', 'Chennai',
       'Durban', 'Abu Dhabi', 'Indore', 'Pune', 'Bengaluru', 'Chandigarh',
       'Cuttack', 'East London', 'Mohali', 'Johannesburg', 'Ranchi',
       'Cape Town', 'Kimberley', 'Bloemfontein']

# pipe = pickle.load(open('pipe.pkl','rb'))
import os

if not os.path.exists('pipe.pkl'):
    st.header("Model file 'pipe.pkl' not found. Please check the path.")
else:
    pipe = pickle.load(open('pipe.pkl', 'rb'))


st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("""
    # **IPL VICTORY PREDICTOR**            
""")
# st.title("IPL Victory Predictor")

col1, col2 = st.columns(2)

with col1:
   
   batting_team =  st.selectbox('Select Batting Team',teams)

with col2:
    if batting_team == '--- select ---':
        bowling_team = st.selectbox('Select Bowling Team', teams)
    else:
        filtered_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox('Select Bowling Team', filtered_teams)

seleted_city = st.selectbox('Select Venue',cities)

target = st.number_input('Target')

col1,col2,col3 = st.columns(3)

with col1:
    score = st.number_input('Score')
with col2:
    overs = st.number_input("Over Completed")
with col3:
    wickets = st.number_input("wicktes down")

if st.button('Predict Winning Probability'):
    try:
        runs_left = target - score
        balls_left = 120 - (overs*6)
        wickets = 10-wickets
        crr = score/overs
        rrr = runs_left/(balls_left/6)
        
    
        input_data = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],
                         'city':[seleted_city],'runs_left':[runs_left],'balls_left':[balls_left],
                         'wickets_remaining':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
        print(input_data)
        st.header("ok till her")

        result = pipe.predict_proba(input_data)

        print(result)
        loss = result[0][0]
        win =  result[0][1]
        st.header(batting_team + " = "+str(round(win*100)) + "%")
        st.header(bowling_team + " = "+str(round(loss*100)) + "%")
    except Exception as e:
        st.header(e, "Some error occured.. Please check you inputs !!")





















#     accuracy = []
# for i in range(101,10001):
#     x_train,x_test,y_train,y_test = tts(x,y,test_size=0.25,random_state=i)
#     trf = ColumnTransformer([
#     ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
# ],
# remainder='passthrough')
#     pipe = Pipeline([
#     ('step1',trf),
#     ('step2',LogisticRegression(solver='liblinear'))
# ])
#     pipe.fit(x_train,y_train)
#     y_pred = pipe.predict(x_test)
#     accuracy.append(accuracy_score(y_test,y_pred))
