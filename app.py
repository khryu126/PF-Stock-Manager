import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- [1. 기본 설정 및 마스터 데이터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_MASTER = {'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8}

# --- [2. 데이터 정제 유틸리티] ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date_smart(series):
    s = series.astype(str).str.replace('.0', '', regex=False).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

def smart_load_csv(file):
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            if df.columns.str.contains('Unnamed').sum() > len(df.columns) * 0.4:
                for i in range(1, 6):
                    file.seek(0)
                    df = pd.read_csv(file, skiprows=i, encoding=enc)
                    if not df.columns.str.contains('Unnamed').all(): break
            return df
        except: continue
    return None

# --- [3. 상세 팝업창 (필터 및 정렬 강화)] ---
@st.dialog("현장별 상세 수주 내역")
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.write(f"🔎 분석 대상 품번: {', '.join(group_ids)}")
    
    code_col = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
    
    # 필터 1: 해당 품번 그룹
    detail = df_bl[df_bl[code_col].astype(str).isin(group_ids)].copy()
    
    # 필터 2: 수주잔량이 0인 현장 제외
    detail = detail[detail['clean_qty'] > 0]
    
    # 필터 3: 설정한 경과 기간 이전의 데이터 제외 (유령 잔량 컷)
    detail = detail[detail['dt_clean'] >= cutoff_date]
    
    if detail.empty:
        st.info("조건에 맞는 수주 데이터가 없습니다.")
        return

    # 필터 4: 납기도래가 가장 빠른 현장 순으로 정렬 (Ascending)
    st.dataframe(detail.sort_values('dt_clean', ascending=True), use_container_width=True, hide_index=True)
    st.caption(f"※ {cutoff_date.strftime('%Y-%m-%d')} 이전의 수주 데이터는 제외되었습니다.")

# --- [4. 메인 UI] ---
st.title("🚀 P·Forecast Stock Manager v3.8")

with st.sidebar:
    st.header("⚙️ 분석 설정")
    start_date = st.date_input("검토 시점(조회 시작일)", datetime.now())
    freq_opt = st.selectbox("집계 단위", ["주별", "월별", "분기별", "년도별"], index=1)
    
    # [추가] N개월 경과 수주잔량 제외 필터
    exclude_months = st.slider("과거 수주 제외 (N개월 경과)", 1, 36, 12, help="조회 시작일 기준 N개월 이전의 수주잔량은 무시합니다.")
    cutoff_date = pd.Timestamp(start_date) - relativedelta(months=exclude_months)
    
    st.markdown("---")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)

# 파일 매핑 로직
data = {}
RECOGNITION = {
    "backlog": ["수주잔량", "총예상수량"], "po": ["PO잔량", "미선적"],
    "stock": ["재고수량", "현재고액"], "item": ["최종생산지명", "이전상품코드"],
    "retail": ["출시예정", "4개월판매량"]
}
if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            cols = "|".join(df.columns)
            for k, v in RECOGNITION.items():
                if any(key in cols for key in v): data[k] = df; break

# --- [5. 메인 분석 로직] ---
if len(data) >= 5:
    with st.spinner('매트릭스를 생성 중입니다...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
        
        # 1. 데이터 정제 및 날짜 엔진
        bl_code = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
        bl_date = '납품예정일' if '납품예정일' in df_bl.columns else df_bl.columns[24]
        df_bl['clean_qty'] = clean_numeric(df_bl['수주잔량'])
        df_bl['dt_clean'] = parse_date_smart(df_bl[bl_date])

        # [필터 적용] 설정된 개월 수 이전의 수주는 분석에서 아예 제외
        df_bl = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

        po_code = '품번' if '품번' in df_po.columns else df_po.columns[12]
        df_po['clean_qty'] = clean_numeric(df_po['PO잔량(미선적)'])
        df_po['dt_clean'] = parse_date_smart(df_po['입고요청일'] if '입고요청일' in df_po.columns else 'PO일자')

        st_code = '품번' if '품번' in df_st.columns else df_st.columns[7]
        st_qty = '재고수량' if '재고수량' in df_st.columns else df_st.columns[17]
        df_st['clean_qty'] = clean_numeric(df_st[st_qty])

        # 2. 기간 축 생성
        base_dt = pd.Timestamp(start_date)
        freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
        date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
        
        time_labels = []
        for i in range(12):
            d = date_range[i]
            if freq_opt == "월별": label = d.strftime('%Y-%m')
            elif freq_opt == "분기별": label = f"{d.year}-{((d.month-1)//3)+1}Q"
            elif freq_opt == "년도별": label = f"{d.year}년"
            else: label = d.strftime('%m/%d')
            time_labels.append(label)

        # 3. 행렬 연산
        target_ids = df_bl[df_bl['clean_qty'] > 0][bl_code].unique()
        matrix_rows = []
        idx = 1

        for pid in target_ids:
            pid_s = str(pid)
            item_match = df_item[df_item['상품코드'].astype(str) == pid_s]
            prev = str(item_match['이전상품코드'].iloc[0]) if not item_match.empty else "-"
            chng = str(item_match['변경상품코드'].iloc[0]) if not item_match.empty else "-"
            group = list(set([pid_s, prev, chng])); group = [g for g in group if g not in ["-", "nan"]]
            
            site = str(item_match['최종생산지명'].iloc[0]) if not item_match.empty else "ETC"
            lt = LT_MASTER.get(site[:2].upper(), 0)

            # 재고 합산
            main_stk = df_st[df_st[st_code].astype(str).isin(group)]['clean_qty'].sum()
            po_kg = df_po[df_po[po_code].astype(str).isin(group)]['clean_qty'].sum()
            po_m = (po_kg * 1000) / (70 * 1.26)

            # 수지 전개
            overdue_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt_clean'] < base_dt)]['clean_qty'].sum()
            running_inv = main_stk - overdue_dem
            
            d_vals, s_vals = {"납기경과": overdue_dem}, {"납기경과": running_inv}

            for i in range(12):
                start, end = date_range[i], date_range[i+1]
                m_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt_clean'] >= start) & (df_bl['dt_clean'] < end)]['clean_qty'].sum()
                m_po_df = df_po[(df_po[po_code].astype(str).isin(group)) & (df_po['dt_clean'] >= start) & (df_po['dt_clean'] < end)]
                m_sup = sum([(r['clean_qty'] * 1000) / (70 * 1.26) for _, r in m_po_df.iterrows()])
                
                running_inv = (running_inv + m_sup) - m_dem
                d_vals[time_labels[i]] = round(m_dem, 0)
                s_vals[time_labels[i]] = round(running_inv, 0)

            # UI 데이터 구성
            common = {"No": idx, "수주품번": pid_s, "본사재고": round(main_stk, 0), "PO잔량(m)": round(po_m, 0), "생산지": f"{site}({lt}M)", "group": group}
            matrix_rows.append({**common, "구분": "소요량", "연계정보": f"이전: {prev}", **d_vals})
            matrix_rows.append({"No": idx, "수주품번": "", "본사재고": "", "PO잔량(m)": "", "생산지": "", "group": group, "구분": "예상재고", "연계정보": f"변경: {chng}", **s_vals})
            idx += 1

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        
        def style_fn(row):
            # 행 번호를 기준으로 2줄씩 묶어 색상 지정 (Row Striping)
            group_idx = (row.name // 2)
            bg_color = '#f5f5f5' if group_idx % 2 == 0 else '#ffffff'
            styles = [f'background-color: {bg_color}'] * len(row)
            
            for i, col in enumerate(row.index):
                # 1. 구분 컬럼은 연한 하늘색 고정
                if col == "구분":
                    styles[i] = 'background-color: #e1f5fe; font-weight: bold'
                # 2. 예상재고 행에서 재고 부족 시 빨간색 표시
                elif row['구분'] == "예상재고" and (col == "납기경과" or col in time_labels):
                    if isinstance(row[col], (int, float)) and row[col] < 0:
                        styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        st.subheader(f"📊 수급 분석 매트릭스 ({freq_opt} 집계)")
        
        sel = st.dataframe(
            res_df.style.apply(style_fn, axis=1),
            use_container_width=True, hide_index=True,
            # 컬럼 순서 조정: 구분 컬럼을 납기경과 바로 왼쪽으로
            column_order=["No", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels,
            on_select="rerun", selection_mode="single-row"
        )

        if sel.selection.rows:
            sel_idx = sel.selection.rows[0]
            if st.button(f"🔍 {res_df.iloc[sel_idx if res_df.iloc[sel_idx]['수주품번'] != '' else sel_idx-1]['수주품번']} 상세 보기"):
                show_detail_popup(res_df.iloc[sel_idx]['group'], df_bl, cutoff_date)
else:
    st.info("사이드바에 5종 파일을 모두 업로드해주세요.")
