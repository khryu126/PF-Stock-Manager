import streamlit as st
import pandas as pd
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as k_image
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

# [1] 설정 및 데이터 로드 (캐싱 적용)
st.set_page_config(page_title="하이브리드 자재 검색기", layout="wide")

@st.cache_resource
def load_models():
    # ResNet50
    model_res = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # DINOv2
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_dino.eval()
    return model_res, model_dino

@st.cache_data
def load_data():
    # 특징 데이터 로드
    with open('material_features.pkl', 'rb') as f:
        db = pickle.load(f)
    
    # 이미지 경로 CSV 로드 (한글 인코딩 CP949 추가)
    try:
        df_path = pd.read_csv('이미지경로.csv', encoding='cp949')
    except UnicodeDecodeError:
        # 만약 파일이 UTF-8로 저장되어 있을 경우를 대비한 안전장치
        df_path = pd.read_csv('이미지경로.csv', encoding='utf-8-sig')
        
    return db, df_path

res_model, dino_model = load_models()
feature_db, path_df = load_data()

# [2] 이미지 전처리 함수
dino_transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# [3] 메인 UI
st.title("🌲 하이브리드 자재 패턴 검색 엔진 (v3.2)")
st.info("ResNet50(결 60%) + DINOv2(구조 40%) 하이브리드 로직이 적용되었습니다.")

uploaded_file = st.file_uploader("📷 찾고 싶은 자재 사진을 업로드하세요", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 사용자 이미지 특징 추출
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="검색 기준 이미지", width=300)
    
    with st.spinner('AI가 패턴의 결(Texture)과 구조(Structure)를 분석 중입니다...'):
        # 1. ResNet50 특징 추출
        x_res = k_image.img_to_array(img.resize((224, 224)))
        res_vec = res_model.predict(preprocess_input(np.expand_dims(x_res, axis=0)), verbose=0).flatten()
        
        # 2. DINOv2 특징 추출
        dino_in = dino_transform(img).unsqueeze(0)
        with torch.no_grad():
            dino_vec = dino_model(dino_in).cpu().numpy().flatten()
            
        # 3. 데이터베이스 내 모든 자재와 비교
        results = []
        for fname, db_vec in feature_db.items():
            # 벡터 분리 (ResNet 2048차원, DINO 384차원)
            db_res = db_vec[:2048]
            db_dino = db_vec[2048:]
            
            # 각각 유사도 계산
            sim_res = cosine_similarity([res_vec], [db_res])[0][0]
            sim_dino = cosine_similarity([dino_vec], [db_dino])[0][0]
            
            # 가중치 합산 (0.6 : 0.4)
            total_sim = (sim_res * 0.6) + (sim_dino * 0.4)
            results.append((fname, total_sim, sim_res, sim_dino))
        
        # 유사도 순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        # [4] 결과 표시
        st.subheader("🔍 가장 유사한 자재 TOP 5")
        cols = st.columns(5)
        
        for i in range(5):
            fname, total_score, s_res, s_dino = results[i]
            
            # CSV에서 구글 드라이브 URL 찾기
            match = path_df[path_df['파일명'] == fname]
            if not match.empty:
                img_url = match.iloc[0]['카카오톡_전송용_URL']
                with cols[i]:
                    st.image(img_url, use_container_width=True)
                    st.write(f"**품번: {fname.split('.')[0]}**")
                    st.write(f"유사도: {total_score:.1%}")
                    st.caption(f"(결 {s_res:.1%}, 구조 {s_dino:.1%})")
            else:
                cols[i].warning(f"경로를 찾을 수 없음: {fname}")

