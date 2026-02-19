import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="성지라미텍 오더 관리 시스템", layout="wide")

# --- 2. 유틸리티 함수 (에러 방지용 안전 설계) ---

def to_num(series):
    """문자열 숫자를 안전하게 실수형으로 변환"""
    if series is None: return pd.Series(0.0)
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.replace(' ', '').str.strip(), errors='coerce').fillna(0.0)

def identify_data(uploaded_files):
    """파일 내용(컬럼명)을 분석해 자동으로 분류"""
    data_map = {}
    for file in uploaded_files:
        identified = False
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            if identified: break
            for sr in [0, 1, 2]: # 최대 2줄 건너뜀
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc, skiprows=sr)
                    df.columns = [str(c).strip() for c in df.columns]
                    cols = " ".join(df.columns)
                    
                    if '수주잔량' in cols and '납품예정일' in cols:
                        data_map['exp'] = df; identified = True; break
                    elif '재고수량' in cols and '현재고액' in cols:
                        data_map['stk'] = df; identified = True; break
                    elif 'PO 수량' in cols or 'PO잔량' in cols:
                        data_map['po'] = df; identified = True; break
                    elif 'B/P무게' in cols or 'B/P weight' in cols:
                        data_map['itm'] = df; identified = True; break
                    elif '4개월판매량' in cols:
                        data_map['rtl'] = df; identified = True; break
                except: continue
    return data_map

# --- 3. 메인 로직 ---

st.title("🛡️ 특판 모양지 통합 오더 관리 시스템 (안정화 버전)")

# 파일 업로드
uploaded_files = st.sidebar.file_uploader("CSV 파일들을 한꺼번에 선택해서 올려주세요", type="csv", accept_multiple_files=True)

if uploaded_files:
    data = identify_data(uploaded_files)
    
    # 필수 파일(수주, 재고) 체크
    if 'exp' in data and 'stk' in data:
        df_exp, df_stk = data['exp'], data['stk']
        df_po, df_itm, df_rtl = data.get('po'), data.get('itm'), data.get('rtl')
        
        # 컬럼 표준화
        exp_col = '상품코드' if '상품코드' in df_exp.columns else '품번'
        stk_col = '품번' if '품번' in df_stk.columns else '상품코드'
        
        # 수주 데이터 전처리 (IndexError 방지)
        df_exp['수주잔량_n'] = to_num(df_exp['수주잔량'])
        df_exp['납기일'] = pd.to_datetime(df_exp['납품예정일'].astype(str), errors='coerce')
        
        # 잔량이 있는 품번만 추출
        active_items = sorted(df_exp[df_exp['수주잔량_n'] > 0][exp_col].unique().tolist())
        
        # 시판 공용 여부 리스트
        retail_list = []
        if df_rtl is not None:
            r_col = '품번' if '품번' in df_rtl.columns else '상품코드'
            retail_list = df_rtl[r_col].unique().tolist()

        # 분석 설정
        unit = st.sidebar.radio("🗓️ 기간 단위", ["월별", "분기별"])
        period_count = st.sidebar.slider("분석 기간", 6, 24, 12)
        
        # 기간 헤더 생성
        now = datetime.now().replace(day=1)
        if unit == "월별":
            periods = [(now + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(period_count)]
        else:
            periods = [f"{(now + pd.DateOffset(months=i*3)).year} Q{((now + pd.DateOffset(months=i*3)).month-1)//3 + 1}" for i in range(period_count // 3)]

        # --- 매트릭스 계산 ---
        matrix_rows = []
        for item in active_items:
            # 품명 및 평량 안전하게 가져오기
            item_exp_data = df_exp[df_exp[exp_col] == item]
            base_name = str(item_exp_data['상품명'].iloc[0]) if not item_exp_data.empty else "알수없음"
            display_name = base_name + " (시판공용)" if item in retail_list else base_name
            
            bw = 70.0
            if df_itm is not None:
                itm_id = '상품코드' if '상품코드' in df_itm.columns else '품번'
                w_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight'
                itm_match = df_itm[df_itm[itm_id] == item]
                if not itm_match.empty:
                    try: bw = float(itm_match[w_col].iloc[0]) or 70.0
                    except: bw = 70.0

            # 초기 가용량 (현재고 + PO)
            inv_m = to_num(df_stk[df_stk[stk_col] == item]['재고수량']).sum()
            if df_po is not None and 'PO 수량' in df_po.columns:
                po_kg = to_num(df_po[df_po['품번'] == item]['PO 수량']).sum()
                inv_m += (po_kg * 1000) / (bw * 1.26)

            # 행 생성 (셀 병합 효과를 위해 아래행 빈칸 처리)
            row_demand = {"품번": item, "상품명": display_name, "구분": "소요량"}
            row_stock = {"품번": "", "상품명": "", "구분": "예상재고"}
            
            current_balance = inv_m
            for p in periods:
                if unit == "월별":
                    p_start = datetime.strptime(p, "%Y-%m")
                    p_end = p_start + pd.DateOffset(months=1)
                else:
                    y, q = int(p.split(' ')[0]), int(p.split('Q')[1])
                    p_start = datetime(y, (q-1)*3 + 1, 1); p_end = p_start + pd.DateOffset(months=3)
                
                # 시판 수요 제외, 순수 특판만 계산
                demand_m = df_exp[(df_exp[exp_col] == item) & (df_exp['납기일'] >= p_start) & (df_exp['납기일'] < p_end)]['수주잔량_n'].sum()
                
                current_balance -= demand_m
                row_demand[p] = int(demand_m)
                row_stock[p] = int(current_balance)
            
            matrix_rows.append(row_demand)
            matrix_rows.append(row_stock)

        # 결과 출력
        final_df = pd.DataFrame(matrix_rows)
        
        # 스타일링 (구형 applymap 사용)
        def color_stock(v):
            if isinstance(v, (int, float)) and v < 0: return 'background-color: #ffcccc; color: #900;'
            if isinstance(v, (int, float)) and v > 0: return 'background-color: #f0fff4; color: #060;'
            return ''

        st.subheader(f"📅 통합 오더 검토 매트릭스 ({unit})")
        st.dataframe(final_df.style.applymap(color_stock, subset=periods), use_container_width=True)
        
        # --- 상세 현장 조회 (안정적인 Selectbox) ---
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            target = st.selectbox("🎯 상세 내역을 볼 품번을 고르세요", active_items)
        
        if target:
            detail = df_exp[df_exp[exp_col] == target][['현장명', '건설사', '수주잔량_n', '납품예정일', '비고']]
            st.table(detail.sort_values('납품예정일'))
            st.caption(f"※ 위 표의 소요량은 '{target}'의 특판 현장 납기 데이터로만 산출되었습니다.")

    else:
        st.warning("⚠️ 필수 파일(수주예정등록, 현재고)이 인식되지 않았습니다. 컬럼명을 확인해 주세요.")
else:
    st.info("👈 왼쪽 사이드바에서 관련 CSV 파일들을 드래그해서 업로드해 주세요.")
