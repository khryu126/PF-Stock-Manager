import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

# --- 1. 페이지 설정 및 스타일 ---
st.set_page_config(page_title="성지라미텍 특판 오더 관리 시스템", layout="wide")

st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    .stDataFrame { border: 1px solid #e6e9ef; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 핵심 유틸리티 함수 ---

def to_num(val):
    """숫자 형식의 문자열(쉼표 포함)을 숫자로 안전하게 변환"""
    if pd.isna(val) or str(val).strip() == '': return 0.0
    try:
        return float(str(val).replace(',', '').strip())
    except:
        return 0.0

def identify_data(uploaded_files):
    """파일명에 상관없이 컬럼명을 분석해 데이터를 자동 분류"""
    data_map = {}
    for file in uploaded_files:
        identified = False
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            if identified: break
            for sr in [0, 1, 2]: # 최대 2줄까지 건너뛰며 헤더 찾기
                try:
                    file.seek(0)
                    temp_df = pd.read_csv(file, encoding=enc, skiprows=sr)
                    temp_df.columns = temp_df.columns.str.strip()
                    cols = " ".join(temp_df.columns.astype(str))
                    
                    # 1. 수주예정등록
                    if '수주잔량' in cols and '납품예정일' in cols:
                        data_map['exp'] = temp_df
                        identified = True; break
                    # 2. 현재고
                    if '재고수량' in cols and '현재고액' in cols:
                        data_map['stk'] = temp_df
                        identified = True; break
                    # 3. PO (발주)
                    if 'PO 수량' in cols and '품번' in cols:
                        data_map['po'] = temp_df
                        identified = True; break
                    # 4. 품목정보 (평량)
                    if 'B/P무게' in cols or 'B/P weight' in cols:
                        data_map['itm'] = temp_df
                        identified = True; break
                    # 5. 시판스펙관리
                    if '4개월판매량' in cols:
                        data_map['rtl'] = temp_df
                        identified = True; break
                except:
                    continue
    return data_map

# --- 3. 메인 로직 ---

st.title("📊 특판 모양지 통합 오더 관리 대시보드")
st.sidebar.header("📁 데이터 업로드")
uploaded_files = st.sidebar.file_uploader("관련 CSV 파일들을 모두 선택해 주세요 (파일명 상관없음)", type="csv", accept_multiple_files=True)

if uploaded_files:
    data = identify_data(uploaded_files)
    
    # 필수 파일 존재 여부 확인
    if 'exp' in data and 'stk' in data:
        df_exp = data['exp']
        df_stk = data['stk']
        df_po = data.get('po')
        df_itm = data.get('itm')
        df_rtl = data.get('rtl')
        
        st.sidebar.success(f"인식 완료: 수주({len(df_exp)}건), 재고({len(df_stk)}건)")
        
        # 분석 단위 및 기간 설정
        st.sidebar.divider()
        unit = st.sidebar.selectbox("🗓️ 분석 단위", ["월별", "분기별"])
        
        # 데이터 정리: 평량 맵 (Basis Weight)
        weight_map = {}
        if df_itm is not None:
            w_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight'
            weight_map = df_itm.set_index('상품코드')[w_col].to_dict()

        # 분석용 품번 추출 (수주잔량이 있는 모든 품번)
        all_items = df_exp[to_num(df_exp['수주잔량']) > 0]['상품코드'].unique()
        
        # 결과 매트릭스 생성
        # 현재부터 12개월간의 기간 생성
        start_date = datetime.now().replace(day=1)
        if unit == "월별":
            periods = [(start_date + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(12)]
        else:
            periods = [f"{(start_date + pd.DateOffset(months=i*3)).year} Q{((start_date + pd.DateOffset(months=i*3)).month-1)//3 + 1}" for i in range(4)]
        
        results = []

        for item in all_items:
            # 1. 기초 정보
            item_name = df_exp[df_exp['상품코드'] == item]['상품명'].iloc[0]
            bw = to_num(weight_map.get(item, 70))
            if bw <= 0: bw = 70.0
            
            # 2. 현재고 (m 단위)
            curr_stock = to_num(df_stk[df_stk['품번'] == item]['재고수량'].sum())
            
            # 3. 입고 예정 (PO -> m 환산)
            po_total = 0
            if df_po is not None:
                po_match = df_po[df_po['품번'] == item]
                po_total = (to_num(po_match['PO 수량'].sum()) * 1000) / (bw * 1.26)
            
            # 4. 시판 평균 수요 (월별)
            rtl_monthly = 0
            if df_rtl is not None:
                rtl_match = df_rtl[df_rtl['품번'] == item]
                if not rtl_match.empty:
                    rtl_monthly = to_num(rtl_match['4개월판매량'].values[0]) / 4

            # 5. 특판 수요 배분 (납품예정일 기준)
            item_exp = df_exp[df_exp['상품코드'] == item].copy()
            item_exp['납기일'] = pd.to_datetime(item_exp['납품예정일'].astype(str), errors='coerce')
            
            # 행 데이터 구성
            row_demand = {"품번": item, "상품명": item_name, "구분": "예상소요량"}
            row_stock = {"품번": item, "상품명": item_name, "구분": "예상재고량"}
            
            running_inv = curr_stock + po_total
            
            for p in periods:
                # 해당 기간 특판 소요량 계산
                if unit == "월별":
                    p_start = pd.to_datetime(p + "-01")
                    p_end = p_start + pd.DateOffset(months=1)
                    monthly_spec = to_num(item_exp[(item_exp['납기일'] >= p_start) & (item_exp['납기일'] < p_end)]['수주잔량'].sum())
                    total_demand = monthly_spec + rtl_monthly
                else:
                    # 분기 계산 (단순화)
                    total_demand = (rtl_monthly * 3) + to_num(item_exp['수주잔량'].sum()) / 12 # 분기별 분산 예시

                running_inv -= total_demand
                row_demand[p] = f"{total_demand:,.0f}"
                row_stock[p] = running_inv
            
            results.append(row_demand)
            results.append(row_stock)

        # 데이터프레임 변환
        final_df = pd.DataFrame(results)

        # 색상 스타일 적용 함수
        def style_matrix(v):
            if isinstance(v, (int, float)):
                if v < 0: return 'background-color: #ffcccc; color: #990000; font-weight: bold;'
                return 'background-color: #e6ffed; color: #006600;'
            return ''

        # 대시보드 출력
        st.subheader(f"📅 {unit} 통합 수지 분석 (향후 리드타임 대응용)")
        st.dataframe(
            final_df.style.applymap(style_matrix, subset=periods),
            use_container_width=True,
            height=600
        )
        
        st.info("💡 빨간색 셀: 재고 부족 시점입니다. 독일 리드타임(4개월)을 고려하여 미리 오더를 진행하세요.")
        
        # 엑셀/CSV 다운로드
        csv = final_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 분석 결과 다운로드 (Excel/CSV)", csv, f"특판_재고분석_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    else:
        st.warning("⚠️ 파일을 찾을 수 없습니다. '수주예정등록'과 '현재고' 컬럼이 포함된 CSV 파일을 업로드해 주세요.")
else:
    st.info("👈 왼쪽 사이드바에서 CSV 파일들을 한꺼번에 드래그해서 올려주세요. (파일명 무관)")
