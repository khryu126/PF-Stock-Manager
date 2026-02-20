import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- [1. 설정 및 마스터 데이터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

# 총 리드타임(LT) 및 선적 리드타임(Shipping LT) 설정
LT_CONFIG = {
    'SE': {'total': 6, 'ship': 3},  # 독일: 총 6개월 / 선적 3개월
    'SRL': {'total': 8, 'ship': 4}, # 이태리: 총 8개월 / 선적 4개월
    'SP': {'total': 8, 'ship': 4},  # 폴란드
    'SH': {'total': 1, 'ship': 0.5},# 상해
    'KD': {'total': 2, 'ship': 1},  # 중국
    'QZ': {'total': 2, 'ship': 1}   # 광저우
}

# --- [2. 유틸리티 함수] ---
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

# --- [3. 상세 팝업창] ---
@st.dialog("현장별 상세 수주 내역", width="large")
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.write(f"🔎 분석 대상 품번 그룹: {', '.join(group_ids)}")
    code_col = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
    detail = df_bl[df_bl[code_col].astype(str).isin(group_ids)].copy()
    detail = detail[(detail['clean_qty'] > 0) & (detail['dt_clean'] >= cutoff_date)]
    if detail.empty:
        st.info("조건에 맞는 수주 데이터가 없습니다.")
        return
    st.dataframe(detail.sort_values('dt_clean', ascending=True), use_container_width=True, hide_index=True)

# --- [4. 메인 UI] ---
st.title("🚀 P·Forecast Stock Manager v4.5")

with st.sidebar:
    st.header("⚙️ 분석 설정")
    start_date = st.date_input("검토 시점(조회 시작일)", datetime.now())
    freq_opt = st.selectbox("집계 단위", ["주별", "월별", "분기별", "년도별"], index=1)
    exclude_months = st.slider("과거 수주 제외 (N개월 경과)", 1, 36, 12)
    cutoff_date = pd.Timestamp(start_date) - relativedelta(months=exclude_months)
    
    st.markdown("---")
    # [추가] 키워드 필터 기능
    search_query = st.text_input("🔍 품명/품번 키워드 검색", "", help="예: Alloy, Oak")
    
    st.markdown("---")
    st.subheader("📁 파일 로드 상태")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)

# 데이터 로드
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

with st.sidebar:
    for k, v in RECOGNITION.items():
        if k in data: st.success(f"✅ {k}")
        else: st.warning(f"⏳ {k} 대기")

# --- [5. 메인 분석 로직] ---
if len(data) >= 5:
    with st.spinner('매트릭스를 생성 중입니다...'):
        df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']
        
        # 1. 데이터 정제 및 날짜 엔진 (선적 LT 반영)
        bl_code = '상품코드' if '상품코드' in df_bl.columns else df_bl.columns[5]
        df_bl['clean_qty'] = clean_numeric(df_bl['수주잔량'])
        df_bl['dt_clean'] = parse_date_smart(df_bl['납품예정일' if '납품예정일' in df_bl.columns else df_bl.columns[24]])
        df_bl = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

        po_code = '품번' if '품번' in df_po.columns else df_po.columns[12]
        df_po['clean_qty'] = clean_numeric(df_po['PO잔량(미선적)'])
        # 생산예정일 기반 입고일 계산 (선적 LT 가산)
        def get_arrival_date(row):
            p_date = parse_date_smart(pd.Series([row.get('생산예정일', np.nan)]))[0]
            if pd.isna(p_date): p_date = parse_date_smart(pd.Series([row.get('입고요청일', row.get('PO일자'))]))[0]
            
            site = str(row.get('생산지명', ''))[:2].upper()
            ship_lt = LT_CONFIG.get(site, {'ship': 0})['ship']
            return p_date + relativedelta(months=int(ship_lt)) if pd.notnull(p_date) else pd.NaT

        df_po['dt_arrival'] = df_po.apply(get_arrival_date, axis=1)

        st_code = '품번' if '품번' in df_st.columns else df_st.columns[7]
        df_st['clean_qty'] = clean_numeric(df_st['재고수량' if '재고수량' in df_st.columns else df_st.columns[17]])

        # 2. 기간 축 및 품번 루프
        base_dt = pd.Timestamp(start_date)
        freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
        date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
        time_labels = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range[:12]]

        target_ids = df_bl[df_bl['clean_qty'] > 0][bl_code].unique()
        matrix_rows = []
        critical_items = [] # 긴급 알람 리스트
        idx = 1

        for pid in target_ids:
            pid_s = str(pid)
            item_match = df_item[df_item['상품코드'].astype(str) == pid_s]
            p_name = str(item_match['상품명'].iloc[0]) if not item_match.empty else "알수없음"
            
            # 필터 기능 적용
            if search_query and (search_query.lower() not in p_name.lower() and search_query.lower() not in pid_s.lower()):
                continue

            prev = str(item_match['이전상품코드'].iloc[0]) if not item_match.empty else ""
            chng = str(item_match['변경상품코드'].iloc[0]) if not item_match.empty else ""
            prev = prev if prev not in ["nan", "0", "-"] else ""
            chng = chng if chng not in ["nan", "0", "-"] else ""

            group = [g for g in [pid_s, prev, chng] if g]
            site = str(item_match['최종생산지명'].iloc[0]) if not item_match.empty else "ETC"
            lt_total = LT_CONFIG.get(site[:2].upper(), {'total': 0})['total']

            is_retail = " 🏷️" if any(str(g) in df_retail.iloc[:, 8].astype(str).values for g in group) else ""
            main_stk = df_st[df_st[st_code].astype(str).isin(group)]['clean_qty'].sum()
            po_kg = df_po[df_po[po_code].astype(str).isin(group)]['clean_qty'].sum()
            po_m = (po_kg * 1000) / (70 * 1.26)

            overdue_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt_clean'] < base_dt)]['clean_qty'].sum()
            running_inv = main_stk - overdue_dem
            d_vals, s_vals = {"납기경과": overdue_dem}, {"납기경과": running_inv}

            is_critical = False
            for i in range(12):
                start, end = date_range[i], date_range[i+1]
                m_dem = df_bl[(df_bl[bl_code].astype(str).isin(group)) & (df_bl['dt_clean'] >= start) & (df_bl['dt_clean'] < end)]['clean_qty'].sum()
                m_sup = sum([(r['clean_qty'] * 1000) / (70 * 1.26) for _, r in df_po[(df_po[po_code].astype(str).isin(group)) & (df_po['dt_arrival'] >= start) & (df_po['dt_arrival'] < end)].iterrows()])
                running_inv = (running_inv + m_sup) - m_dem
                d_vals[time_labels[i]], s_vals[time_labels[i]] = round(m_dem, 0), round(running_inv, 0)
                
                # 리드타임 내 재고 부족 판정 (알람용)
                if running_inv < 0 and start < base_dt + relativedelta(months=lt_total):
                    is_critical = True

            if is_critical: critical_items.append(f"{p_name} ({pid_s})")

            common = {"No": idx, "품명": p_name, "수주품번": pid_s + is_retail, "본사재고": main_stk, "PO잔량(m)": po_m, "생산지": f"{site}({lt_total}M)", "group": group}
            matrix_rows.append({**common, "구분": "소요량", "연계정보": f"이전:{prev}" if prev else "", **d_vals})
            matrix_rows.append({"No": idx, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "group": group, "구분": "예상재고", "연계정보": f"변경:{chng}" if chng else "", **s_vals})
            idx += 1

    # [추가] 상단 긴급 알람 섹션
    if critical_items:
        with st.expander(f"⚠️ 긴급 발주 검토 대상 ({len(critical_items)}건)", expanded=False):
            st.error("아래 품목은 생산 리드타임 이내에 재고 고갈이 예상됩니다.")
            st.write(", ".join(critical_items))

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        def style_fn(row):
            g_idx = (row.name // 2)
            base_bg = '#f5f5f5' if g_idx % 2 == 0 else '#ffffff'
            styles = [f'background-color: {base_bg}'] * len(row)
            for i, col in enumerate(row.index):
                if col == "구분": styles[i] = 'background-color: #e1f5fe; font-weight: bold'
                elif row['구분'] == "예상재고" and (col == "납기경과" or col in time_labels):
                    if row[col] < 0: styles[i] = 'background-color: #ff4b4b; color: white'
            return styles

        sel = st.dataframe(
            res_df.style.apply(style_fn, axis=1).format({"본사재고": "{:,.0f}", "PO잔량(m)": "{:,.0f}"}, na_rep=""),
            use_container_width=True, hide_index=True,
            column_order=["No", "품명", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels,
            on_select="rerun", selection_mode="single-row"
        )
        if sel.selection.rows:
            sel_idx = sel.selection.rows[0]
            target = res_df.iloc[sel_idx if res_df.iloc[sel_idx]['수주품번'] != '' else sel_idx-1]
            if st.button(f"🔍 {target['수주품번']} 상세 현황"): show_detail_popup(target['group'], df_bl, cutoff_date)
