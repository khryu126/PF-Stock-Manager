import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# 앱 제목 및 설정
st.set_page_config(page_title="자재 패턴 검색기", page_icon="🔍")
st.title("🔍 실시간 자재 패턴 검색")
st.write("현장에서 찍은 사진을 올리면 가장 유사한 자재를 찾아드립니다.")

# 1. 데이터 로드 (캐싱을 통해 속도 향상)
@st.cache_resource
def load_resources():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # 장부(.pkl)와 엑셀 파일을 앱과 같은 폴더에 두면 됩니다.
    with open('자재_지문_장부_light.pkl', 'rb') as f:
        feature_dict = pickle.load(f)
    
    spec_df = pd.read_csv('스펙인코드_25.12.08.csv', encoding='cp949')
    link_df = pd.read_csv('제목 없는 스프레드시트 - 시트1.csv', encoding='cp949')
    return model, feature_dict, spec_df, link_df

try:
    model, feature_dict, spec_df, link_df = load_resources()
except:
    st.error("데이터 파일을 불러오지 못했습니다. 파일 위치를 확인해주세요.")

# 2. 사진 업로드 섹션
uploaded_file = st.file_uploader("가구 사진을 촬영하거나 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 업로드한 사진 보여주기
    img = Image.open(uploaded_file)
    st.image(img, caption='업로드된 사진', use_column_width=True)
    
    with st.spinner('유사한 자재를 분석 중입니다...'):
        # AI 분석
        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        target_feat = model.predict(x).flatten()
        
        # 대조 작업
        scores = [(f, cosine_similarity([target_feat], [feat])[0][0]) for f, feat in feature_dict.items()]
        top_results = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        
        st.subheader("✨ 분석 결과 Top 3")
        for i, (fname, score) in enumerate(top_results):
            m = link_df[link_df['파일명'] == fname]
            if not m.empty:
                pumbun = m.iloc[0]['추출된_품번']
                url = m.iloc[0]['카카오톡_전송용_URL']
                s = spec_df[spec_df['품번'] == str(pumbun).strip()]
                name = s.iloc[0]['품명'] if not s.empty else "정보없음"
                
                # 결과 카드 형태 출력
                with st.expander(f"{i+1}순위: {name} (일치율 {score*100:.1f}%)"):
                    st.write(f"**품번:** {pumbun}")
                    st.link_button("구글 드라이브 사진 확인", url)