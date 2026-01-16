import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import os
import ssl
import torch
import torchvision.transforms as T
import cv2
import base64
from PIL import Image, ImageEnhance, ImageDraw
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as k_image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# [0] 환경 설정: SSL 우회 (DINOv2 다운로드 에러 방지)
ssl._create_default_https_context = ssl._create_unverified_context

# --- [1] 유틸리티 및 리소스 로드 ---
def load_csv_smart(target_name):
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    st.error(f"❌ {target_name} 파일을 찾을 수 없습니다.")
    st.stop()

def get_digits(text):
    return "".join(re.findall(r'\d+', str(text))) if text else ""

@st.cache_resource
def init_resources():
    # 모델 1: ResNet50 (결/텍스처 분석)
    model_res = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # 모델 2: DINOv2 (구조/패턴 분석)
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_dino.eval()
    
    # 데이터 로드
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
        
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv')
    
    # 재고 로직 (유 대리님 기존 로직 유지)
    df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
    agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
    stock_date = str(int(df_stock['정산일자'].max())) if '정산일자' in df_stock.columns else "확인불가"
    
    return model_res, model_dino, feature_db, df_path, df_info, agg_stock, stock_date

res_model, dino_model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

# [DINOv2 전용 변환]
dino_transform = T.Compose([
    T.Resize(224), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_data
def get_master_map():
    mapping = {}
    for _, row in df_info.iterrows():
        f = str(row.get('상품코드', '')).strip()
        n = str(row.get('상품명', '')).strip()
        d = get_digits(f)
        if d: mapping[d] = {'formal': f, 'name': n}
    return mapping

master_map = get_master_map()

# --- [2] 이미지 처리 엔진 (Perspective & Filters) ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    w1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    w2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    h1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    h2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    mW, mH = max(int(w1), int(w2)), max(int(h1), int(h2))
    dst = np.array([[0, 0], [mW - 1, 0], [mW - 1, mH - 1], [0, mH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (mW, mH))

def apply_smart_filters(img, category, lighting, brightness, sharpness):
    if lighting == '백열등 (누런 조명)':
        r, g, b = img.split(); b = b.point(lambda i: i * 1.2); img = Image.merge('RGB', (r, g, b))
    en_con = ImageEnhance.Contrast(img); en_shp = ImageEnhance.Sharpness(img); en_bri = ImageEnhance.Brightness(img)
    if category != '일반':
        img = en_shp.enhance(2.0); img = en_con.enhance(1.1)
    if brightness != 1.0: img = en_bri.enhance(brightness)
    if sharpness != 1.0: img = en_shp.enhance(sharpness)
    return img

# --- [3] 메인 UI 레이아웃 ---
st.set_page_config(layout="wide", page_title="하이브리드 자재 검색 v3.3")
st.title("🌲 하이브리드 자재 패턴 검색 엔진")
st.sidebar.info(f"📅 재고 기준일: {stock_date}")

if 'points' not in st.session_state: st.session_state['points'] = []
if 'search_done' not in st.session_state: st.session_state['search_done'] = False

uploaded = st.file_uploader("📸 분석할 자재 사진을 업로드하세요", type=['jpg','png','jpeg'])

if uploaded:
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
        st.session_state['points'] = []; st.session_state['search_done'] = False
        st.session_state['current_img_name'] = uploaded.name
        st.session_state['proc_img'] = Image.open(uploaded).convert('RGB')
        st.rerun()

    working_img = st.session_state['proc_img']
    w, h = working_img.size
    
    # 보기 크기 조절
    scale = st.radio("🔍 보기 크기:", [0.3, 0.5, 0.7, 1.0], format_func=lambda x: f"{int(x*100)}%", index=1, horizontal=True)
    
    col_opt, col_pad = st.columns([1, 2])
    with col_opt:
        mat_type = st.selectbox("🧱 자재 종류", ['일반', '마루/우드 (Wood)', '하이그로시/유광 (Glossy)', '벽지/패브릭 (Texture)'])
        s_mode = st.radio("🔎 검색 모드", ["종합 검색(6:4)", "패턴 중심(흑백)"], horizontal=True)
        bri = st.slider("밝기", 0.5, 2.0, 1.0, 0.1)
        shp = st.slider("선명도", 0.0, 3.0, 1.5, 0.1)
        if st.button("❌ 선택 초기화"): st.session_state['points'] = []; st.rerun()

    with col_pad:
        d_img = working_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(d_img)
        for i, p in enumerate(st.session_state['points']):
            px, py = p[0]*scale, p[1]*scale
            draw.ellipse((px-8, py-8, px+8, py+8), fill='red', outline='white')
        
        value = streamlit_image_coordinates(d_img, key="coords")
        if value and len(st.session_state['points']) < 4:
            new_p = (value['x']/scale, value['y']/scale)
            if not st.session_state['points'] or st.session_state['points'][-1] != new_p:
                st.session_state['points'].append(new_p); st.rerun()

    if len(st.session_state['points']) == 4:
        warped = four_point_transform(np.array(working_img), np.array(st.session_state['points'], dtype="float32"))
        final_img = Image.fromarray(warped)
        final_img = apply_smart_filters(final_img, mat_type, '일반', bri, shp)
        if s_mode == "패턴 중심(흑백)": final_img = final_img.convert("L").convert("RGB")
        
        st.image(final_img, width=300, caption="분석 영역")
        
        if st.button("🔍 하이브리드 검색 시작", type="primary", use_container_width=True):
            with st.spinner('결(ResNet)과 구조(DINO)를 6:4 비율로 분석 중...'):
                # 1. 사용자 이미지 특징 추출 (Hybrid)
                x_res = k_image.img_to_array(final_img.resize((224, 224)))
                q_res = res_model.predict(preprocess_input(np.expand_dims(x_res, axis=0)), verbose=0).flatten()
                
                d_in = dino_transform(final_img).unsqueeze(0)
                with torch.no_grad():
                    q_dino = dino_model(d_in).cpu().numpy().flatten()

                # 2. 유사도 계산 (0.6:0.4 가중치 합산)
                all_results = []
                for fn, db_vec in feature_db.items():
                    db_res = db_vec[:2048]
                    db_dino = db_vec[2048:]
                    
                    s_res = cosine_similarity([q_res], [db_res])[0][0]
                    s_dino = cosine_similarity([q_dino], [db_dino])[0][0]
                    total_sim = (s_res * 0.6) + (s_dino * 0.4)
                    
                    # 정보 매칭
                    info = master_map.get(get_digits(fn), {'formal': fn, 'name': '정보 없음'})
                    qty = agg_stock.get(info['formal'].strip().upper(), 0)
                    url_row = df_path[df_path['추출된_품번'].apply(get_digits) == get_digits(fn)]
                    url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                    
                    if url:
                        all_results.append({'formal': info['formal'], 'name': info['name'], 'score': total_sim, 'stock': qty, 'url': url})

                all_results.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['search_results'] = all_results[:20]
                st.session_state['search_done'] = True; st.rerun()

# --- [4] 결과 출력 (구글 드라이브 이미지 직접 연결) ---
if st.session_state.get('search_done'):
    st.markdown("---")
    res = st.session_state['search_results']
    cols = st.columns(5)
    for i, item in enumerate(res):
        with cols[i % 5]:
            # [힌트 적용] 6464번 라인의 직접 URL 방식을 사용하여 엑박 방지 및 속도 향상
            st.image(item['url'], use_container_width=True)
            st.markdown(f"**{item['formal']}**")
            st.caption(f"{item['name']} ({item['score']:.1%})")
            st.info(f"재고: {item['stock']:,}m")
