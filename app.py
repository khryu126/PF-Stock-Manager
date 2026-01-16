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
import requests
import base64
from PIL import Image, ImageEnhance, ImageDraw
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as k_image
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_image_coordinates import streamlit_image_coordinates

# [0] 환경 설정 및 보안 대응
ssl._create_default_https_context = ssl._create_unverified_context

# --- [1] 유틸리티 함수 ---
def get_direct_url(url):
    if not url or str(url) == 'nan' or 'drive.google.com' not in url: return url
    if 'file/d/' in url: file_id = url.split('file/d/')[1].split('/')[0]
    elif 'id=' in url: file_id = url.split('id=')[1].split('&')[0]
    else: return url
    return f'https://drive.google.com/uc?export=download&id={file_id}'

def get_image_as_base64(url):
    try:
        r = requests.get(get_direct_url(url), timeout=10)
        img_str = base64.b64encode(r.content).decode()
        return f"data:image/png;base64,{img_str}"
    except: return None

def load_csv_smart(target_name):
    files = os.listdir('.')
    for f in files:
        if f.lower() == target_name.lower():
            for enc in ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']:
                try: return pd.read_csv(f, encoding=enc)
                except: continue
    return pd.DataFrame()

def get_digits(text):
    return "".join(re.findall(r'\d+', str(text))) if text else ""

# --- [2] 리소스 로딩 (캐싱) ---
@st.cache_resource
def init_resources():
    model_res = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_dino.eval()
    
    with open('material_features.pkl', 'rb') as f:
        feature_db = pickle.load(f)
        
    df_path = load_csv_smart('이미지경로.csv')
    df_info = load_csv_smart('품목정보.csv')
    df_stock = load_csv_smart('현재고.csv') # 단가 열이 없어도 문제없이 로드됨
    
    agg_stock, stock_date = {}, "확인불가"
    if not df_stock.empty:
        # v2.6 기반 정밀 재고 로직
        df_stock['재고수량'] = pd.to_numeric(df_stock['재고수량'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df_stock['품번_KEY'] = df_stock['품번'].astype(str).str.strip().str.upper()
        agg_stock = df_stock.groupby('품번_KEY')['재고수량'].sum().to_dict()
        if '정산일자' in df_stock.columns:
            stock_date = str(int(df_stock['정산일자'].max()))
            
    return model_res, model_dino, feature_db, df_path, df_info, agg_stock, stock_date

res_model, dino_model, feature_db, df_path, df_info, agg_stock, stock_date = init_resources()

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

# --- [3] 이미지 보정 및 변환 엔진 ---
def apply_advanced_correction(img, angle, bri, con, shp, sat, temp, exp, hue):
    if angle != 0: img = img.rotate(angle, expand=True)
    img = ImageEnhance.Brightness(img).enhance(bri)
    img = ImageEnhance.Contrast(img).enhance(con)
    img = ImageEnhance.Sharpness(img).enhance(shp)
    img = ImageEnhance.Color(img).enhance(sat)
    
    img_np = np.array(img).astype(np.float32)
    img_np *= exp
    if temp > 1.0: img_np[:, :, 0] *= temp; img_np[:, :, 2] /= temp
    elif temp < 1.0: img_np[:, :, 2] *= (2.0-temp); img_np[:, :, 0] /= (2.0-temp)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    if hue != 0:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
        img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_np)

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    w = max(int(np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))), int(np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))))
    h = max(int(np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))), int(np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LANCZOS4)

# --- [4] Deco Finder UI ---
st.set_page_config(layout="wide", page_title="Deco Finder - Schattdecor")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #B67741; color: white; border-radius: 4px; }
    .stExpander { border: 1px solid #B67741; border-radius: 5px; }
    h1 { color: #B67741; font-family: 'Arial Black', sans-serif; margin-bottom: 0px; }
    .stock-tag { font-weight: bold; padding: 2px 6px; border-radius: 3px; font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True)

# 상단 로고 (Logo.png 파일 사용) 및 타이틀
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists("Logo.png"):
        st.image("Logo.png", width=120)
    else:
        st.image("https://brandfetch.com/schattdecor.com?view=library", width=120) # 파일 없을 시 대비용
with col_title:
    st.title("Deco Finder")
    st.caption("AI-Powered Surface Matching & Inventory Solution")

st.sidebar.markdown(f"📦 **재고 정산 기준일:** \n{stock_date}")

if 'points' not in st.session_state: st.session_state['points'] = []
if 'search_done' not in st.session_state: st.session_state['search_done'] = False
if 'refresh_count' not in st.session_state: st.session_state['refresh_count'] = 0

uploaded = st.file_uploader("📸 자재 사진 업로드 (Upload Image)", type=['jpg','png','jpeg'])

if uploaded:
    if 'current_img_name' not in st.session_state or st.session_state['current_img_name'] != uploaded.name:
        st.session_state.update({'points': [], 'search_done': False, 'current_img_name': uploaded.name, 'proc_img': Image.open(uploaded).convert('RGB')})
        st.rerun()

    working_img = st.session_state['proc_img']
    w, h = working_img.size

    # 1. 고급 보정 옵션 (한글 메인)
    with st.expander("🛠️ 고급 이미지 보정 및 회전 (Advanced Correction)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            angle = st.slider("사진 회전 (Rotation)", 0, 360, 0)
            bri = st.slider("밝기 (Brightness)", 0.5, 2.0, 1.0)
            con = st.slider("대비 (Contrast)", 0.5, 2.0, 1.0)
        with c2:
            sat = st.slider("채도 (Saturation)", 0.0, 2.0, 1.0)
            shp = st.slider("선명도 (Sharpness)", 0.0, 3.0, 1.5)
            exp = st.slider("노출 (Exposure)", 0.5, 2.0, 1.0)
        with c3:
            temp = st.slider("색온도 (Color Temp)", 0.5, 1.5, 1.0)
            hue = st.slider("색조 (Hue Shift)", 0, 180, 0)
            if st.button("🔄 선택 영역 초기화"): st.session_state['points'] = []; st.rerun()

    # 2. 영역 지정
    scale = st.radio("🔍 보기 크기 (View Scale):", [0.1, 0.3, 0.5, 0.7, 1.0], index=2, horizontal=True)
    
    col_ui, col_pad = st.columns([1, 2])
    with col_ui:
        source_type = st.radio("자재 출처", ['📸 실물 촬영', '💻 디지털 샘플'], horizontal=True)
        mat_type = st.selectbox("자재 분류", ['일반(Normal)', '우드(Wood)', '하이그로시(Glossy)', '패브릭(Texture)', '석재(Stone)'])
        s_mode = st.radio("분석 모드", ["종합 검색(6:4)", "패턴 중심(흑백)"], horizontal=True)
        if st.button("🔄 이미지 새로고침"): st.session_state['refresh_count'] += 1; st.rerun()

    with col_pad:
        d_img = working_img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(d_img)
        for i, p in enumerate(st.session_state['points']):
            px, py = p[0]*scale, p[1]*scale
            draw.ellipse((px-8, py-8, px+8, py+8), fill='#B67741', outline='white', width=2)
            draw.text((px+10, py-10), str(i+1), fill='red')
        
        if len(st.session_state['points']) == 4:
            draw.polygon([tuple((p[0]*scale, p[1]*scale)) for p in st.session_state['points']], outline='#00FF00', width=3)

        coords = streamlit_image_coordinates(d_img, key=f"deco_{st.session_state['refresh_count']}")
        if coords and len(st.session_state['points']) < 4:
            new_p = (coords['x']/scale, coords['y']/scale)
            if not st.session_state['points'] or st.session_state['points'][-1] != new_p:
                st.session_state['points'].append(new_p); st.rerun()

    if len(st.session_state['points']) == 4:
        warped = four_point_transform(np.array(working_img), np.array(st.session_state['points'], dtype="float32"))
        final_img = Image.fromarray(warped)
        final_img = apply_advanced_correction(final_img, angle, bri, con, shp, sat, temp, exp, hue)
        if s_mode == "패턴 중심(흑백)": final_img = final_img.convert("L").convert("RGB")
        
        st.image(final_img, width=300, caption="Deco Finder 분석 대상")
        
        if st.button("🔍 Deco Finder 검색 시작", type="primary", use_container_width=True):
            with st.spinner('하이브리드 엔진이 질감과 구조를 분석 중입니다...'):
                x_res = k_image.img_to_array(final_img.resize((224, 224)))
                q_res = res_model.predict(preprocess_input(np.expand_dims(x_res, axis=0)), verbose=0).flatten()
                d_in = dino_transform(final_img).unsqueeze(0)
                with torch.no_grad(): q_dino = dino_model(d_in).cpu().numpy().flatten()

                results = []
                for fn, db_vec in feature_db.items():
                    score = (cosine_similarity([q_res], [db_vec[:2048]])[0][0] * 0.6) + \
                            (cosine_similarity([q_dino], [db_vec[2048:]])[0][0] * 0.4)
                    
                    d_key = get_digits(fn)
                    info = master_map.get(d_key, {'formal': fn.split('.')[0], 'name': '정보 없음'})
                    
                    # [v2.6 이식] 정밀 매칭 키 적용
                    f_key = str(info['formal']).strip().upper()
                    qty = agg_stock.get(f_key, 0)
                    
                    url_row = df_path[df_path['추출된_품번'].apply(get_digits) == d_key]
                    url = url_row['카카오톡_전송용_URL'].values[0] if not url_row.empty else None
                    if url:
                        results.append({'formal': info['formal'], 'name': info['name'], 'score': score, 'url': url, 'stock': qty})

                results.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['search_results'] = results[:15]
                st.session_state['search_done'] = True; st.rerun()

# --- [5] 결과 출력 (탭 시스템 및 정보 노출 강화) ---
if st.session_state.get('search_done'):
    st.markdown("---")
    res_all = st.session_state['search_results']
    res_stock = [r for r in res_all if r['stock'] > 0] # 재고 보유분 필터링

    tab1, tab2 = st.tabs(["📊 전체 결과 (Total)", "✅ 재고 보유 (In-Stock)"])

    def display_grid(items):
        if not items:
            st.warning("해당하는 자재가 없습니다.")
            return
        cols = st.columns(5)
        for i, item in enumerate(items):
            with cols[i % 5]:
                st.markdown(f"**{i+1}위: {item['formal']}**")
                
                # 펼치기 전 재고 노출
                if item['stock'] >= 100:
                    st.markdown(f"<span class='stock-tag' style='color:#155724; background-color:#d4edda;'>재고: {item['stock']:,}m</span>", unsafe_allow_html=True)
                elif item['stock'] > 0:
                    st.markdown(f"<span class='stock-tag' style='color:#856404; background-color:#fff3cd;'>재고: {item['stock']:,}m</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='stock-tag' style='color:#721c24; background-color:#f8d7da;'>재고 없음</span>", unsafe_allow_html=True)
                
                st.caption(f"유사도: {item['score']:.1%}")
                
                with st.expander("🖼️ 상세보기 (Details)", expanded=False):
                    b64 = get_image_as_base64(item['url'])
                    if b64: st.image(b64, use_container_width=True)
                    st.write(f"**품명:** {item['name']}")

    with tab1: display_grid(res_all)
    with tab2: display_grid(res_stock)
