import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# --- 1. 페이지 설정 및 디자인 ---
st.set_page_config(page_title="성지라미텍 특판 오더 관리 시스템 V4", layout="wide")

st.markdown("""
    <style>
    .stDataFrame { border: 1px solid #e6e9ef; }
    .reportview-container .main .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 유틸리티 함수 ---

def to_num_series(series):
    if series is None or series.empty: return pd.Series(0.0)
    s = series.astype(str).str.replace(',', '').str.replace(' ', '').str.strip()
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def identify_data(uploaded_files):
    data_map = {}
    for file in uploaded_files:
        identified = False
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            if identified: break
            for sr in [0, 1]:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc, skiprows=sr)
                    df.columns = df.columns.astype(str).str.strip()
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

# --- 3. 메인 화면 ---

st.title("📊 특판 모양지 통합 오더 관리 (V4: 시판공용 식별)")
st.sidebar.info("수주 데이터 기반으로만 소요량을 계산합니다. 시판 공용 여부는 상품명에 표시됩니다.")

files = st.sidebar.file_uploader("CSV 파일들을 한꺼번에 드래그하여 업로드하세요", type="csv", accept_multiple_files=True)

if files:
    data = identify_data(files)
    
    if 'exp' in data and 'stk' in data:
        df_exp, df_stk = data['exp'], data['stk']
        df_po, df_itm, df_rtl = data.get('po'), data.get('itm'), data.get('rtl')

        # 컬럼 표준화
        exp_col = '상품코드' if '상품코드' in df_exp.columns else '품번'
        stk_col = '품번' if '품번' in df_stk.columns else '상품코드'
        
        # 데이터 정제
        df_exp['납기일'] = pd.to_datetime(df_exp['납품예정일'].astype(str), errors='coerce')
        df_exp['수주잔량_n'] = to_num_series(df_exp['수주잔량'])
        
        # 수주잔량 있는 품번만 추출
        active_items = df_exp[df_exp['수주잔량_n'] > 0][exp_col].unique()
        
        # 시판 공용 품번 리스트 추출 (시판스펙관리 파일이 있을 경우)
        retail_item_list = []
        if df_rtl is not None:
            rtl_col = '품번' if '품번' in df_rtl.columns else '상품코드'
            retail_item_list = df_rtl[rtl_col].unique().tolist()

        unit = st.sidebar.radio("🗓️ 분석 단위", ["월별", "분기별"])
        months_to_show = st.sidebar.slider("분석 기간", 6, 24, 12)
        
        # 기간 헤더 생성
        now = datetime.now().replace(day=1)
        if unit == "월별":
            periods = [(now + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(months_to_show)]
        else:
            periods = [f"{(now + pd.DateOffset(months=i*3)).year} Q{((now + pd.DateOffset(months=i*3)).month-1)//3 + 1}" for i in range(months_to_show // 3)]

        matrix_rows = []
        for item in active_items:
            item_info = df_exp[df_exp[exp_col] == item]
            base_name = item_info['상품명'].iloc[0] if not item_info.empty else "알수없음"
            
            # 시판 공용 표시 추가
            display_name = base_name + " 🏷️(시판공용)" if item in retail_item_list else base_name
            
            # 평량 및 재고 계산
            bw = 70.0
            if df_itm is not None:
                itm_id = '상품코드' if '상품코드' in df_itm.columns else '품번'
                w_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight'
                bw_val = df_itm[df_itm[itm_id] == item][w_col].iloc[0] if item in df_itm[itm_id].values else 70.0
                try: bw = float(bw_val) if float(bw_val) > 0 else 70.0
                except: bw = 70.0

            curr_m = to_num_series(df_stk[df_stk[stk_col] == item]['재고수량']).sum()
            po_m = 0
            if df_po is not None and '품번' in df_po.columns:
                po_kg = to_num_series(df_po[df_po['품번'] == item]['PO 수량']).sum()
                po_m = (po_kg * 1000) / (bw * 1.26)

            # 행 생성 (시각적 셀병합 효과)
            row_c = {"품번": item, "상품명": display_name, "구분": "소요량"}
            row_s = {"품번": "", "상품명": "", "구분": "예상재고"} 
            
            balance = curr_m + po_m
            for p in periods:
                if unit == "월별":
                    p_start = datetime.strptime(p, "%Y-%m")
                    p_end = p_start + pd.DateOffset(months=1)
                    # 시판 수요 0으로 설정 (사용자 요청 반영)
                    spec_m = df_exp[(df_exp[exp_col] == item) & (df_exp['납기일'] >= p_start) & (df_exp['납기일'] < p_end)]['수주잔량_n'].sum()
                    total_demand = spec_m
                else:
                    y, q = int(p.split(' ')[0]), int(p.split('Q')[1])
                    p_start = datetime(y, (q-1)*3 + 1, 1); p_end = p_start + pd.DateOffset(months=3)
                    spec_m = df_exp[(df_exp[exp_col] == item) & (df_exp['납기일'] >= p_start) & (df_exp['납기일'] < p_end)]['수주잔량_n'].sum()
                    total_demand = spec_m

                balance -= total_demand
                row_c[p] = round(total_demand)
                row_s[p] = round(balance)
            
            matrix_rows.append(row_c); matrix_rows.append(row_s)

        final_df = pd.DataFrame(matrix_rows)

        # 스타일링
        def style_fn(v):
            if isinstance(v, (int, float)) and v < 0: return 'background-color: #ffcccc; color: #900; font-weight: bold;'
            if isinstance(v, (int, float)) and v > 0: return 'background-color: #f0fff4; color: #060;'
            return ''

        st.subheader("🗓️ 특판 수주잔량 기반 오더 시점 검토")
        selected = st.dataframe(
            final_df.style.applymap(style_fn, subset=periods),
            use_container_width=True, height=550,
            on_select="rerun", selection_mode="single_row"
        )

        # --- 상세 현장 내역 ---
        if len(selected.selection.rows) > 0:
            idx = selected.selection.rows[0]
            sel_item = final_df.iloc[idx]['품번'] if final_df.iloc[idx]['품번'] != "" else final_df.iloc[idx-1]['품번']
            
            st.divider()
            st.subheader(f"🔍 [{sel_item}] 현장별 납기 상세 내역")
            detail_df = df_exp[df_exp[exp_col] == sel_item][['현장명', '건설사', '수주잔량_n', '납품예정일', '비고']]
            st.table(detail_df.sort_values(by='납품예정일'))
            st.caption("※ 이 표는 시판 수요를 제외한 **순수 특판 수주 데이터**로만 계산되었습니다.")

    else:
        st.warning("⚠️ 필수 파일(수주예정등록, 현재고)을 먼저 올려주세요.")
else:
    st.info("👈 사이드바에 파일을 드래그하여 업로드하세요. (시판스펙관리.csv 포함 시 공용 여부가 표시됩니다.)")
