import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- [1. 설정 및 리드타임 마스터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_CONFIG = {
    'SE': {'total': 6, 'ship_days': 90},
    'SR': {'total': 8, 'ship_days': 90},
    'SRL': {'total': 8, 'ship_days': 90},
    'SP': {'total': 8, 'ship_days': 90},
    'SH': {'total': 1, 'ship_days': 30},
    'KD': {'total': 2, 'ship_days': 30},
    'QZ': {'total': 2, 'ship_days': 30}
}

# --- [2. 유틸리티 함수] ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(r'[^\d.-]', '', regex=True)
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date_smart(series):
    s = series.astype(str).str.replace('.0', '', regex=False).str.strip()
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

def find_col_precise(df, keywords, exclude_keywords=None, default_idx=None):
    for k in keywords:
        for col in df.columns:
            col_upper = str(col).replace(" ", "").upper()
            if k in col_upper:
                if exclude_keywords:
                    if any(ex.upper() in col_upper for ex in exclude_keywords): continue
                return col
    if default_idx is not None and len(df.columns) > default_idx:
        return df.columns[default_idx]
    return None

def smart_load_csv(file):
    for enc in ['cp949', 'utf-8-sig', 'utf-8']:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            if df.columns.str.contains('Unnamed').sum() > len(df.columns) * 0.3:
                for i in range(1, 20):
                    file.seek(0)
                    df = pd.read_csv(file, skiprows=i, encoding=enc)
                    if not df.columns.str.contains('Unnamed').all(): break
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except: continue
    return None

# --- [3. 상세 수주 팝업 (잔량 0 제외 및 시급순 정렬)] ---
@st.dialog("상세 수주 내역", width="large")
def show_detail_popup(group_ids, df_bl, cutoff_date):
    st.markdown(f"#### 🔍 분석 대상 품번 그룹: `{', '.join(group_ids)}`")
    
    code_col = find_col_precise(df_bl, ['상품코드', '품번'], default_idx=5)
    qty_col = find_col_precise(df_bl, ['수주잔량', '잔량'], default_idx=30)
    
    group_upper = [g.upper() for g in group_ids]
    detail = df_bl[df_bl[code_col].astype(str).str.upper().str.strip().isin(group_upper)].copy()
    detail['clean_qty'] = clean_numeric(detail[qty_col])
    
    # 납기일자 파싱 (인덱스 24)
    detail['dt_clean_popup'] = pd.to_datetime(detail.iloc[:, 24].astype(str).str.replace('.0',''), format='%Y%m%d', errors='coerce')
    
    # [요청사항] 수주잔량 0 제외 및 납기 가장 빠른 순(오름차순) 정렬
    detail = detail[(detail['clean_qty'] > 0) & (detail['dt_clean_popup'] >= cutoff_date)]
    
    if detail.empty:
        st.info("조건에 맞는 미출고 수주 데이터가 없습니다.")
        return
        
    st.dataframe(
        detail.sort_values('dt_clean_popup', ascending=True), 
        use_container_width=True, hide_index=True
    )

# --- [4. 메인 UI 및 사이드바 (정의 필수)] ---
st.title("🚀 P·Forecast Stock Manager v6.8")

RECOGNITION = {
    "backlog": {"name": "수주예정(Demand)", "keys": ["수주잔량", "총예상수량"], "found": False},
    "po": {"name": "구매발주(PO)", "keys": ["PO잔량", "미선적"], "found": False},
    "stock": {"name": "현재고(Stock)", "keys": ["재고수량", "현재고액"], "found": False},
    "item": {"name": "품목정보(Master)", "keys": ["최종생산지명", "이전상품코드"], "found": False},
    "retail": {"name": "시판스펙(Retail)", "keys": ["출시예정", "4개월판매량"], "found": False}
}

with st.sidebar:
    st.header("⚙️ 분석 설정")
    default_start = (datetime.now().replace(day=1) + relativedelta(months=1))
    start_date_val = st.date_input("검토 시점(조회 시작일)", default_start)
    freq_opt = st.selectbox("집계 단위", ["주별", "월별", "분기별", "년도별"], index=1)
    exclude_months = st.slider("과거 수주 제외 (N개월)", 1, 36, 12)
    cutoff_date = pd.Timestamp(start_date_val) - relativedelta(months=exclude_months)
    st.markdown("---")
    search_query = st.text_input("🔍 품명/품번 키워드 검색", "")
    st.markdown("---")
    uploaded_files = st.file_uploader("5종 CSV 파일 업로드", accept_multiple_files=True)
    
    # [v6.8 추가] 분석 실행 버튼 - 누르기 전까지는 (1/227) 로딩이 걸리지 않음
    run_button = st.button("📊 수급 분석 시작/갱신", type="primary", use_container_width=True)

# 파일 로드 및 분류
data_files = {}
if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            cols_text = "|".join(df.columns).upper()
            for k, v in RECOGNITION.items():
                if any(key in cols_text for key in v["keys"]):
                    data_files[k] = df; RECOGNITION[k]["found"] = True; break

# --- [5. 시뮬레이션 엔진] ---
def run_simulation():
    # 진행률 표시줄 생성
    progress_bar = st.progress(0, text="데이터 분석 중...")
    
    df_item, df_bl, df_po, df_st, df_retail = data_files['item'], data_files['backlog'], data_files['po'], data_files['stock'], data_files['retail']
    today_dt = pd.Timestamp(datetime.now().date())
    base_dt = pd.Timestamp(start_date_val)

    # 마스터 정보 구축 (최신순)
    it_code = find_col_precise(df_item, ['상품코드', '품번'], exclude_keywords=['대표'], default_idx=6)
    it_site = find_col_precise(df_item, ['최종생산지명', '생산지'], default_idx=12)
    it_prev = find_col_precise(df_item, ['이전상품코드'], default_idx=13)
    it_date = find_col_precise(df_item, ['생성일자'], default_idx=3)
    it_name = find_col_precise(df_item, ['상품명', '품명'], default_idx=1)

    master_proc = df_item.copy()
    master_proc['clean_date'] = parse_date_smart(master_proc[it_date])
    master_proc['key_u'] = master_proc[it_code].astype(str).str.upper().str.strip()
    master_proc = master_proc.sort_values(by=['key_u', 'clean_date'], ascending=[True, False])
    master_unique = master_proc.drop_duplicates(subset='key_u', keep='first')

    site_map = master_unique.set_index('key_u')[it_site].to_dict()
    prev_map = master_unique.set_index('key_u')[it_prev].to_dict()
    next_map = master_unique.set_index(master_unique[it_prev].astype(str).str.upper().str.strip())[it_code].to_dict()

    # 소스 데이터 정제
    bl_code_col = find_col_precise(df_bl, ['상품코드', '품번'], default_idx=5)
    df_bl['clean_qty'] = clean_numeric(df_bl[find_col_precise(df_bl, ['수주잔량', '총예상수량'], default_idx=30)])
    df_bl['dt_clean'] = parse_date_smart(df_bl[find_col_precise(df_bl, ['납품예정일'], default_idx=24)])
    df_bl_filtered = df_bl[df_bl['dt_clean'] >= cutoff_date].copy()

    po_code_col = find_col_precise(df_po, ['품번', '상품코드'], default_idx=12)
    df_po['m_qty'] = clean_numeric(df_po[find_col_precise(df_po, ['PO잔량', '미선적'], default_idx=19)]) * 11.3378 

    # 리드타임 및 입고일 계산
    def calc_arrival(row):
        pid_u = str(row[po_code_col]).upper().strip()
        site_v = str(row.get(find_col_precise(df_po, ['생산지명', '거래처'], default_idx=10), site_map.get(pid_u, 'ETC'))).upper()
        site_k = 'SR' if 'SR' in site_v else site_v[:2]
        lt = LT_CONFIG.get(site_k, LT_CONFIG.get(site_v[:2], {'total': 1, 'ship_days': 30}))
        p_dt = parse_date_smart(pd.Series([row.get(find_col_precise(df_po, ['생산예정일'], default_idx=28), np.nan)]))[0]
        if pd.notnull(p_dt): arrival = p_dt + pd.DateOffset(days=int(lt['ship_days']))
        else:
            b_dt = parse_date_smart(pd.Series([row.get(find_col_precise(df_po, ['PO일자', '발주일자'], default_idx=3), today_dt)]))[0]
            if pd.isna(b_dt): b_dt = today_dt
            arrival = b_dt + pd.DateOffset(months=int(lt['total']))
        if pd.isnull(arrival) or arrival < base_dt:
            arrival = today_dt + pd.DateOffset(days=int(lt['ship_days']))
            if arrival < base_dt: arrival = base_dt
        return arrival

    df_po['dt_arrival'] = df_po.apply(calc_arrival, axis=1)
    df_st['clean_qty'] = clean_numeric(df_st[find_col_precise(df_st, ['재고수량', '현재고'], default_idx=7)])

    # 날짜 범위 설정
    freq_map = {"주별": "W", "월별": "MS", "분기별": "QS", "년도별": "YS"}
    date_range = pd.date_range(start=base_dt, periods=13, freq=freq_map[freq_opt])
    time_labels = [d.strftime('%Y-%m-%d' if freq_opt=="주별" else '%Y-%m') for d in date_range[:12]]

    target_ids = df_bl_filtered[df_bl_filtered['clean_qty'] > 0][bl_code_col].unique()
    matrix_rows, alert_list = [], []
    
    for i, pid in enumerate(target_ids):
        pid_s = str(pid).strip(); pid_u = pid_s.upper()
        item_match = master_unique[master_unique['key_u'] == pid_u]
        p_name = str(item_match[it_name].iloc[0]) if not item_match.empty else "-"
        
        # [v6.8] 로딩 게이지 업데이트
        progress_bar.progress((i + 1) / len(target_ids), text=f"🔍 분석 중 ({i+1}/{len(target_ids)}): {p_name[:10]}...")

        def clean_p(v):
            s = str(v).strip().upper()
            return s if s not in ["NAN", "NONE", "0", "-", ""] else ""
        p_id = clean_p(prev_map.get(pid_u, "")); n_id = clean_p(next_map.get(pid_u, ""))
        group = list(set([pid_u, p_id, n_id])); group = [g for g in group if g]

        site_name = str(site_map.get(pid_u, "ETC"))
        site_key = 'SR' if 'SR' in site_name.upper() else site_name[:2].upper()
        lt_total = LT_CONFIG.get(site_key, {'total': 0})['total']

        main_stk = df_st[df_st[find_col_precise(df_st, ['품번', '상품코드'], default_idx=7)].astype(str).str.upper().str.strip().isin(group)]['clean_qty'].sum()
        overdue_dem = df_bl_filtered[(df_bl_filtered[bl_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_bl_filtered['dt_clean'] < base_dt)]['clean_qty'].sum()
        running_inv = main_stk - overdue_dem
        
        d_row = {"No": i+1, "품명": p_name, "수주품번": pid_s, "본사재고": main_stk, "PO잔량(m)": df_po[df_po[po_code_col].astype(str).str.upper().str.strip().isin(group)]['m_qty'].sum(), "생산지": f"{site_key}({lt_total}M)", "구분": "소요량", "연계정보": f"이전:{p_id}" if p_id else "", "납기경과": overdue_dem, "group": group}
        p_row = {"No": i+1, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "입고량(PO)", "연계정보": "", "납기경과": 0, "group": group}
        s_row = {"No": i+1, "품명": "", "수주품번": "", "본사재고": np.nan, "PO잔량(m)": np.nan, "생산지": "", "구분": "예상재고", "연계정보": f"변경:{n_id}" if n_id else "", "납기경과": running_inv, "group": group}

        for j in range(12):
            start, end = date_range[j], date_range[j+1]
            m_dem = df_bl_filtered[(df_bl_filtered[bl_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_bl_filtered['dt_clean'] >= start) & (df_bl_filtered['dt_clean'] < end)]['clean_qty'].sum()
            m_sup = df_po[(df_po[po_code_col].astype(str).str.upper().str.strip().isin(group)) & (df_po['dt_arrival'] >= start) & (df_po['dt_arrival'] < end)]['m_qty'].sum()
            running_inv = (running_inv + m_sup) - m_dem
            lbl = time_labels[j]
            d_row[lbl], p_row[lbl], s_row[lbl] = m_dem, m_sup, running_inv
            if running_inv < 0 and start < base_dt + pd.DateOffset(months=lt_total):
                alert_list.append({"품명": p_name, "품번": pid_s, "생산지": site_key, "LT": lt_total, "부족시점": lbl, "부족수량": abs(running_inv), "group": group})
        matrix_rows.extend([d_row, p_row, s_row])

    progress_bar.empty()
    return pd.DataFrame(matrix_rows), pd.DataFrame(alert_list), time_labels

# --- [6. 메인 로직 제어] ---
if len(data_files) >= 5:
    # 분석 데이터가 세션에 없거나 '분석 시작' 버튼을 눌렀을 때만 시뮬레이션 실행
    if 'sim_data' not in st.session_state or run_button:
        res, alerts, labels = run_simulation()
        st.session_state.sim_data = {'res': res, 'alerts': alerts, 'labels': labels}

    res_df = st.session_state.sim_data['res']
    alert_df = st.session_state.sim_data['alerts']
    time_labels = st.session_state.sim_data['labels']

    # 1. 긴급 발주 대시보드 (최상단 노출)
    st.subheader("🚨 수급 안정성 검토 (긴급 품목)")
    if not alert_df.empty:
        alert_clean = alert_df.drop_duplicates(subset=['품번'], keep='first').copy()
        
        def get_dday(row):
            deadline = pd.to_datetime(row['부족시점']) - pd.DateOffset(months=int(row['LT']))
            days = (deadline - pd.Timestamp(datetime.now().date())).days
            return f"D-{days}일" if days >= 0 else f"지남({abs(days)}일 전)"
        
        alert_clean['발주기한'] = alert_clean.apply(get_dday, axis=1)
        
        # 긴급 리스트 클릭 시 상세보기 연동
        sel_alert = st.dataframe(
            alert_clean[['품명', '품번', '생산지', '부족시점', '부족수량', '발주기한']], 
            use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
        )
        if sel_alert.selection.rows:
            target = alert_clean.iloc[sel_alert.selection.rows[0]]
            # [v6.8 핵심] 누르면 바로 해당 품번 팝업 노출
            if st.button(f"🔍 {target['품번']} 상세보기 (긴급 리스트용)", type="primary"):
                show_detail_popup(target['group'], data_files['backlog'], cutoff_date)
    else:
        st.success("안전: 리드타임 내 부족 예정 품목이 없습니다.")

    # 2. 메인 시뮬레이션 매트릭스
    st.subheader(f"📊 통합 수급 분석 매트릭스")
    if search_query:
        res_df = res_df[res_df['품명'].str.contains(search_query, case=False) | res_df['수주품번'].str.contains(search_query, case=False)]

    num_cols = ["본사재고", "PO잔량(m)", "납기경과"] + time_labels
    def style_fn(row):
        g_idx = (row.name // 3); bg = '#f9f9f9' if g_idx % 2 == 0 else '#ffffff'
        styles = [f'background-color: {bg}'] * len(row)
        for i, col in enumerate(row.index):
            if col == "구분": styles[i] = 'background-color: #e1f5fe; font-weight: bold'
            elif row['구분'] == "예상재고" and col in num_cols and row[col] < 0:
                styles[i] = 'background-color: #ff4b4b; color: white'
        return styles

    st_df = st.dataframe(
        res_df.style.apply(style_fn, axis=1).format({c: "{:,.0f}" for c in num_cols}, na_rep=""),
        use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row",
        column_order=["No", "품명", "수주품번", "본사재고", "PO잔량(m)", "생산지", "연계정보", "구분", "납기경과"] + time_labels
    )
    
    if st_df.selection.rows:
        target = res_df.iloc[st_df.selection.rows[0] - (st_df.selection.rows[0] % 3)]
        if st.button(f"🔍 {str(target['수주품번']).strip()} 상세 내역 보기"):
            show_detail_popup(target['group'], data_files['backlog'], cutoff_date)
else:
    st.info("사이드바에 5종 파일을 모두 업로드해주세요.")
