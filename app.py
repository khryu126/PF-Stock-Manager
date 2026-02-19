import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, utils
import re
from datetime import datetime, timedelta

# --- 1. 페이지 설정 및 디자인 ---
st.set_page_config(page_title="성지라미텍 특판 리스크 관리 시스템", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 유틸리티 함수 (데이터 로드 및 정제) ---

def safe_read_csv(file, skiprows=0):
    """다양한 인코딩 대응 및 파일 로드 안전장치"""
    if file is not None:
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, skiprows=skiprows)
                df.columns = df.columns.str.strip() # 컬럼명 공백 제거
                return df
            except:
                continue
    return None

def to_numeric(series):
    """문자열 숫자를 계산 가능한 숫자로 변환 (쉼표 제거 등)"""
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)

def clean_site_name(name):
    """현장명에서 불필요한 노이즈 제거 (특판 필터링 최적화)"""
    if not name or pd.isna(name): return ""
    # 1. 특수문자 및 수식어 제거
    name = re.sub(r'\(주\)|주식회사|신축공사|현장|일대|M/H|MH|S/H|SH|샘플', '', str(name))
    # 2. 공백 정제
    name = " ".join(name.split())
    return name

# --- 3. 사이드바: 데이터 업로드 ---
st.sidebar.header("📁 데이터 소스 업로드")
st.sidebar.info("CSV 파일을 순서대로 업로드해 주세요.")

f_expected = st.sidebar.file_uploader("1. 수주예정등록.csv (첫 줄 공백 포함)", type="csv")
f_stock = st.sidebar.file_uploader("2. 현재고.csv", type="csv")
f_history = st.sidebar.file_uploader("3. 출고내역.csv", type="csv")
f_item = st.sidebar.file_uploader("4. 품목정보.csv", type="csv")
f_retail = st.sidebar.file_uploader("5. 시판스펙관리.csv", type="csv")
f_po = st.sidebar.file_uploader("6. PO.csv", type="csv")

# 데이터 로드
df_exp = safe_read_csv(f_expected, skiprows=1)
df_stk = safe_read_csv(f_stock)
df_his = safe_read_csv(f_history)
df_itm = safe_read_csv(f_item)
df_rtl = safe_read_csv(f_retail)
df_po = safe_read_csv(f_po)

# --- 4. 메인 화면 구성 ---
st.title("🛡️ 성지라미텍 특판 리스크 관리 시스템")

# 필수 파일 체크
if df_exp is not None and df_stk is not None:
    # 전처리: 수치 데이터 변환
    df_exp['수주잔량_n'] = to_numeric(df_exp['수주잔량'])
    df_stk['재고수량_n'] = to_numeric(df_stk['재고수량'])
    
    # 평량(Basis Weight) 맵 구축 (품목정보 파일이 있을 때만)
    weight_map = {}
    if df_itm is not None:
        # 컬럼명 유연하게 대응 (한글/영문)
        w_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight' if 'B/P weight' in df_itm.columns else None
        if w_col:
            weight_map = df_itm.set_index('상품코드')[w_col].to_dict()

    tab1, tab2 = st.tabs(["📍 현장 누락 방지 점검", "📅 오더 시점 및 재고 예측"])

    # --- TAB 1: 현장 누락 방지 ---
    with tab1:
        st.subheader("🏢 특판 현장(M/H, S/H) 출고 기반 등록 여부 확인")
        
        if df_his is not None:
            # 특판 중요 키워드 필터링 (샘플 등 자질구레한 건 제외)
            target_keywords = ['M/H', 'MH', 'S/H', 'SH']
            mh_pattern = '|'.join(target_keywords)
            
            mh_deliveries = df_his[
                df_his['현장명'].str.contains(mh_pattern, na=False, case=False) |
                df_his['비고'].str.contains(mh_pattern, na=False, case=False)
            ].copy()

            if not mh_deliveries.empty:
                # 고유 현장명 추출 및 정제
                unique_sites = mh_deliveries['현장명'].unique()
                expected_sites = df_exp['현장명'].unique()
                
                # 매칭 속도 향상을 위해 정제된 리스트 미리 생성
                clean_exp_list = [clean_site_name(s) for s in expected_sites]
                exp_map = {clean_site_name(s): s for s in expected_sites}

                results = []
                for site in unique_sites:
                    c_site = clean_site_name(site)
                    # 유사도 매칭 (RapidFuzz)
                    match = process.extractOne(c_site, clean_exp_list, processor=utils.default_process)
                    score = match[1] if match else 0
                    match_original = exp_map.get(match[0]) if match else "없음"
                    
                    status = "✅ 등록됨" if score > 85 else "⚠️ 누락 의심" if score > 50 else "🔴 미등록"
                    results.append({
                        "출고 현장명(원문)": site,
                        "정제 후 이름": c_site,
                        "가장 유사한 수주명": match_original,
                        "신뢰도": f"{score:.1f}%",
                        "상태": status
                    })
                
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                st.info("💡 '미등록'이나 '누락 의심'으로 뜨는 건은 AI 검색 기능을 통해 주소지 정보를 대조해볼 필요가 있습니다.")
            else:
                st.write("M/H 또는 S/H 키워드가 포함된 출고 내역이 없습니다.")
        else:
            st.warning("출고내역.csv 파일을 업로드해 주세요.")

    # --- TAB 2: 오더 시점 및 재고 예측 ---
    with tab2:
        st.subheader("모양지 발주 검토 (독일 리드타임 4개월)")
        
        target_item = st.selectbox("분석할 품번(상품코드)을 선택하세요", df_exp['상품코드'].unique())
        
        # 1. 현재고
        curr_inv = df_stk[df_stk['품번'] == target_item]['재고수량_n'].sum()
        
        # 2. PO 잔량 환산 (kg -> m)
        po_m = 0
        if df_po is not None:
            po_data = df_po[df_po['품번'] == target_item].copy()
            if not po_data.empty:
                bw = weight_map.get(target_item, 70) # 평량 없으면 기본 70g
                # 환산 공식: m = (kg * 1000) / (평량 * 1.26)
                po_m = (to_numeric(po_data['PO 수량']).sum() * 1000) / (bw * 1.26)
        
        # 3. 수요 집계
        spec_demand = df_exp[df_exp['상품코드'] == target_item]['수주잔량_n'].sum()
        retail_monthly = 0
        if df_rtl is not None:
            rtl_match = df_rtl[df_rtl['품번'] == target_item]
            if not rtl_match.empty:
                retail_monthly = to_numeric(rtl_match['4개월판매량']).values[0] / 4

        # 대시보드 지표
        c1, c2, c3 = st.columns(3)
        c1.metric("현재고 (m)", f"{curr_inv:,.0f}")
        c2.metric("PO 예정량 (m)", f"{po_m:,.0f}")
        c3.metric("특판 수주잔량 (m)", f"{spec_demand:,.0f}")

        # 그래프 시뮬레이션
        st.write("### 📉 향후 6개월 재고 시뮬레이션 (시판 수요 포함)")
        months = [(datetime.now() + timedelta(days=30*i)).strftime("%Y-%m") for i in range(1, 7)]
        sim_balance = curr_inv + po_m
        graph_data = []
        for m in months:
            sim_balance -= retail_monthly # 시판 수요 매달 차감
            graph_data.append({"월": m, "예상재고": max(0, sim_balance)})
        
        st.line_chart(pd.DataFrame(graph_data).set_index("월"))
        
        if sim_balance < spec_demand:
            st.error(f"🚨 위험: 4개월 내 재고 쇼트 발생 가능성 높음! (부족분: {spec_demand - sim_balance:,.0f} m)")
            st.warning("독일 수입 리드타임을 고려하여 발주 시점을 점검하세요.")
        else:
            st.success("안정권: 현재 가용량으로 특판 수주 물량 대응이 가능합니다.")

else:
    st.warning("사이드바에서 '수주예정등록.csv'와 '현재고.csv'를 먼저 업로드해 주세요.")
