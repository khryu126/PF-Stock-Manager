import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="성지라미텍 특판 오더 집중 관리", layout="wide")

# --- 2. 유틸리티 함수 (숫자 변환 및 파일 식별) ---

def to_num_series(series):
    """표 전체를 숫자로 안전하게 변환 (쉼표 제거 포함)"""
    if series is None or series.empty:
        return pd.Series(0.0)
    s = series.astype(str).str.replace(',', '').str.replace(' ', '').str.strip()
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def identify_data(uploaded_files):
    """파일명 무관하게 컬럼명을 분석해 데이터 자동 분류"""
    data_map = {}
    for file in uploaded_files:
        identified = False
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            if identified: break
            for sr in [0, 1, 2]: # 최대 2줄 건너뜀
                try:
                    file.seek(0)
                    temp_df = pd.read_csv(file, encoding=enc, skiprows=sr)
                    temp_df.columns = temp_df.columns.astype(str).str.strip()
                    cols = " ".join(temp_df.columns)
                    
                    # 식별 로직
                    if '수주잔량' in cols and '납품예정일' in cols:
                        data_map['exp'] = temp_df
                        identified = True; break
                    elif '재고수량' in cols and '현재고액' in cols:
                        data_map['stk'] = temp_df
                        identified = True; break
                    elif 'PO 수량' in cols or 'PO잔량' in cols:
                        data_map['po'] = temp_df
                        identified = True; break
                    elif 'B/P무게' in cols or 'B/P weight' in cols:
                        data_map['itm'] = temp_df
                        identified = True; break
                    elif '4개월판매량' in cols:
                        data_map['rtl'] = temp_df
                        identified = True; break
                except:
                    continue
    return data_map

# --- 3. 메인 화면 ---

st.title("📦 특판 모양지 오더 집중 관리 대시보드")
st.sidebar.header("📁 데이터 통합 업로드")
files = st.sidebar.file_uploader("관련 CSV 파일들을 모두 선택해 주세요", type="csv", accept_multiple_files=True)

if files:
    data = identify_data(files)
    
    # 필수 파일(수주, 재고) 체크
    if 'exp' in data and 'stk' in data:
        df_exp = data['exp']
        df_stk = data['stk']
        df_po = data.get('po')
        df_itm = data.get('itm')
        df_rtl = data.get('rtl')
        
        # 컬럼 표준화
        exp_item_col = '상품코드' if '상품코드' in df_exp.columns else '품번'
        stk_item_col = '품번' if '품번' in df_stk.columns else '상품코드'
        
        # 1단계: 유 대리님 요청대로 '수주잔량이 0보다 큰' 품번만 필터링
        df_exp['수주잔량_n'] = to_num_series(df_exp['수주잔량'])
        active_items = df_exp[df_exp['수주잔량_n'] > 0][exp_item_col].unique()
        
        st.sidebar.success(f"분석 대상: {len(active_items)}개 품번 (잔량 보유 건)")
        
        # 분석 설정
        unit = st.sidebar.radio("🗓️ 분석 단위", ["월별", "분기별"])
        period_count = st.sidebar.slider("분석 기간", 6, 24, 12)
        
        # 평량 맵 구축
        weight_map = {}
        if df_itm is not None:
            itm_id = '상품코드' if '상품코드' in df_itm.columns else '품번'
            w_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight'
            weight_map = df_itm.set_index(itm_id)[w_col].to_dict()

        # 기간 생성
        start_date = datetime.now().replace(day=1)
        if unit == "월별":
            periods = [(start_date + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(period_count)]
        else:
            periods = [f"{(start_date + pd.DateOffset(months=i*3)).year} Q{((start_date + pd.DateOffset(months=i*3)).month-1)//3 + 1}" for i in range(period_count // 3)]
        
        matrix_rows = []

        for item in active_items:
            # 품명 및 기본 정보
            item_df = df_exp[df_exp[exp_item_col] == item]
            item_name = item_df['상품명'].iloc[0] if not item_df.empty and '상품명' in item_df.columns else "이름없음"
            
            bw = weight_map.get(item, 70.0)
            try: bw = float(bw) if float(bw) > 0 else 70.0
            except: bw = 70.0
            
            # 현재 가용량 (현재고 + PO)
            inv_m = to_num_series(df_stk[df_stk[stk_item_col] == item]['재고수량']).sum()
            if df_po is not None and 'PO 수량' in df_po.columns:
                po_kg = to_num_series(df_po[df_po['품번'] == item]['PO 수량']).sum()
                inv_m += (po_kg * 1000) / (bw * 1.26)
            
            # 시판 월 수요
            rtl_m = 0
            if df_rtl is not None and '4개월판매량' in df_rtl.columns:
                rtl_m = to_num_series(df_rtl[df_rtl['품번'] == item]['4개월판매량']).sum() / 4
            
            # 특판 납기 배분
            item_exp = df_exp[df_exp[exp_item_col] == item].copy()
            item_exp['date'] = pd.to_datetime(item_exp['납품예정일'].astype(str), errors='coerce')
            
            row_demand = {"품번": item, "상품명": item_name, "구분": "소요량(m)"}
            row_stock = {"품번": item, "상품명": item_name, "구분": "예상재고(m)"}
            
            balance = inv_m
            for p in periods:
                if unit == "월별":
                    p_start = datetime.strptime(p, "%Y-%m")
                    p_end = p_start + pd.DateOffset(months=1)
                    spec_m = item_exp[(item_exp['date'] >= p_start) & (item_exp['date'] < p_end)]['수주잔량_n'].sum()
                    total_m = spec_m + rtl_m
                else:
                    y, q = int(p.split(' ')[0]), int(p.split('Q')[1])
                    p_start = datetime(y, (q-1)*3 + 1, 1)
                    p_end = p_start + pd.DateOffset(months=3)
                    spec_m = item_exp[(item_exp['date'] >= p_start) & (item_exp['date'] < p_end)]['수주잔량_n'].sum()
                    total_m = spec_m + (rtl_m * 3)
                
                balance -= total_m
                row_demand[p] = round(total_m)
                row_stock[p] = round(balance)
            
            matrix_rows.append(row_demand)
            matrix_rows.append(row_stock)

        # 결과 렌더링
        final_df = pd.DataFrame(matrix_rows)

        def style_fn(val):
            if isinstance(val, (int, float, np.integer, np.floating)):
                if val < 0: return 'background-color: #ffcccc; color: #990000; font-weight: bold;'
                return 'background-color: #e6ffed; color: #006600;'
            return ''

        st.subheader(f"📅 품번별 {unit} 오더 검토 매트릭스")
        st.dataframe(final_df.style.applymap(style_fn, subset=periods), use_container_width=True, height=600)
        
        st.info("💡 빨간색 칸: 재고 고갈 시점입니다. 최소 4개월 전(독일 리드타임)에 오더를 검토하세요.")
        csv = final_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 결과 다운로드 (CSV)", csv, f"order_plan_{datetime.now().strftime('%m%d')}.csv", "text/csv")

    else:
        st.warning("⚠️ '수주예정등록'과 '현재고' 파일이 필요합니다. 컬럼명을 확인해 주세요.")
else:
    st.info("👈 왼쪽에서 분석할 파일들을 한꺼번에 올려주세요.")
