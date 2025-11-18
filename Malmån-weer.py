import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime
import re 
# NIEUWE IMPORTS VOOR API CALL
import requests
from io import StringIO

# 1. Configuratie en constanten (Constants and Configuration)

# ===============================================================================
# Dit is de basis-URL voor de map /weatherdata/ op GitHub.
# ===============================================================================
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Pillmaster/ericmeteo/main/weatherdata/"

# Het jaar waar de data in de repository begint. (AANDACHTSPUNT: Controleer of dit uw oudste jaar is)
START_YEAR = 2025

# Tijdzone definitie voor Stockholm (inclusief DST/CEST en CET)
TARGET_TIMEZONE = 'Europe/Stockholm'

# Mapping van Stations-ID naar gebruiksvriendelijke namen
STATION_MAP = {
    '2308LH047': 'Malmån hus',
    '2102LH011': 'Malmån sjön'
}

# Lijst van numerieke kolommen die gevisualiseerd kunnen worden
NUMERIC_COLS = ['battery', 'dauwpunt', 'luchtvocht', 'druk', 'zoninstraling', 'temp', 'natbol']

# Vriendelijke namen voor kolommen in de UI en grafieken
COL_DISPLAY_MAP = {
    'battery': 'Batterijspanning (V)',
    'dauwpunt': 'Dauwpunt (°C)', 
    'luchtvocht': 'Luchtvochtigheid (%)',
    'druk': 'Luchtdruk (hPa)', 
    'zoninstraling': 'Zoninstraling (W/m²)',
    'temp': 'Temperatuur (°C)',
    'natbol': 'Natteboltemperatuur (°C)'
}

# Omgekeerde mapping voor het ophalen van de originele kolomnamen
DISPLAY_TO_COL_MAP = {display_name: col_name for col_name, display_name in COL_DISPLAY_MAP.items()}

# NIEUW: Definitie van de 30-jarige Klimaatnormaalperioden (Jaar-Jaar)
# De waarden zijn (Startdatum YYYY-MM-DD, Einddatum YYYY-MM-DD)
CLIMATE_NORMAL_PERIODS = {
    "1990-2019 (Standaard WMO Normaal)": ("1990-01-01", "2019-12-31"),
    "1980-2009": ("1980-01-01", "2009-12-31"),
    "1970-1999": ("1970-01-01", "1999-12-31"),
    "1960-1989": ("1960-01-01", "1989-12-31"),
    "1950-1979": ("1950-01-01", "1979-12-31"),
    "1940-1969": ("1940-01-01", "1969-12-31"),
}

# NIEUW: Bepaling van de totale periode voor één API-oproep (om 429 errors te voorkomen)
BENCHMARK_START_DATE_FULL = "1940-01-01"
BENCHMARK_END_DATE_FULL = "2019-12-31" 


# 2. Functies voor Data (Loading & Processing)

# Zoekt naar beschikbare jaren op GitHub (Gecached, NU ZONDER TTL VOOR ACTUEELHEID)
@st.cache_data(show_spinner="Zoeken naar beschikbare jaren op GitHub...")
def discover_available_years(start_year, station_id, github_base_url):
    """
    Probeert iteratief om weather_YYYY.csv te laden van het startjaar tot het huidige jaar.
    """
    current_year = datetime.datetime.now().year
    available_years = []
    
    for year in range(start_year, current_year + 1):
        full_url = f"{github_base_url}{station_id}/weather_{year}.csv"
        
        try:
            # Laad alleen de header en de eerste rij (nrows=1) voor een snelle check
            df_test = pd.read_csv(full_url, sep=';', on_bad_lines='skip', nrows=1)
            
            # Controleer of het DataFrame niet leeg is en de verwachte kolom bevat
            if not df_test.empty and 'datum_waarneming_UTC' in df_test.columns:
                available_years.append(year)
            
        except Exception:
            pass
            
    return sorted(available_years)


# Bestaande Laadfunctie (caching behouden voor performantie)
@st.cache_data
def load_data(station_id, years, github_base_url, station_map, target_timezone):
    """
    Laadt, parseert en pre-verwerkt weerdata van GitHub voor meerdere jaren.
    """
    all_years_data = []
    station_name = station_map.get(station_id, station_id)

    for year in years:
        full_url = f"{github_base_url}{station_id}/weather_{year}.csv"

        try:
            df = pd.read_csv(full_url, sep=';', on_bad_lines='skip')

            df['Timestamp_UTC_str'] = df['datum_waarneming_UTC'] + ' ' + df['tijd_waarneming_UTC']
            df['Timestamp_UTC'] = pd.to_datetime(df['Timestamp_UTC_str'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
            df = df.dropna(subset=['Timestamp_UTC'])

            for col in NUMERIC_COLS:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'druk' in df.columns:
                df['druk'] = df['druk'] / 100 

            df['Timestamp_UTC'] = df['Timestamp_UTC'].dt.tz_localize('UTC') 
            df['Timestamp_Local'] = df['Timestamp_UTC'].dt.tz_convert(target_timezone)
            
            all_years_data.append(df)
            
        except Exception as e:
            st.warning(f"❌ Bestand niet gevonden of fout bij laden voor {station_name} in {year}. Reden: {e}")

    if all_years_data:
        df_station = pd.concat(all_years_data, ignore_index=True)
        return df_station.sort_values('Timestamp_Local').copy() 
    else:
        return pd.DataFrame() 

# -------------------------------------------------------------------
# NIEUWE FUNCTIE: Ophalen COMPLETE Externe Historische Data via Open-Meteo API (Grote Cache)
# Wordt eenmaal aangeroepen om alle benchmark data te verzamelen.
# -------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner="Laden complete historische benchmark data (1940-2019)...") 
def fetch_complete_historical_data(start_date_str, end_date_str):
    """
    Haalt de volledige reeks historische data op in één keer om rate limits te vermijden.
    """
    LAT = 62.9977
    LON = 17.0811
    
    api_url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date_str, 
        "end_date": end_date_str,     
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
        "timezone": "Europe/Stockholm",
        "format": "csv"
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status() 
        
        csv_data = response.text.split('\n', 3)[3] 
        df_hist = pd.read_csv(StringIO(csv_data))
        
        df_hist.columns = df_hist.columns.str.strip()
        
        df_hist = df_hist.rename(columns={
            'time': 'Date',
            'temperature_2m_max (°C)': 'Temp_High_C', 
            'temperature_2m_min (°C)': 'Temp_Low_C',  
            'temperature_2m_mean (°C)': 'Temp_Avg_C', 
        })
        
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        
        success_message = f"Complete historische data ({start_date_str[:4]}-{end_date_str[:4]}) van {len(df_hist)} dagen succesvol geladen via Open-Meteo (ERA5)."
        return df_hist, ('success', success_message)
        
    except requests.exceptions.HTTPError as e:
        error_message = f"❌ Kon historische data niet ophalen via API. Fout: {e}. Probeer later opnieuw of herlaad de app."
        return pd.DataFrame(), ('error', error_message)
    except Exception as e:
        error_message = f"⚠️ Algemene fout bij het verwerken van API-data. Fout: {e}"
        return pd.DataFrame(), ('warning', error_message)

# -------------------------------------------------------------------
# GECORRIGEERDE FUNCTIE: Ophalen Externe Historische Data via Open-Meteo API (voor Enkelvoudige Benchmark)
# Roept de complete set op en filtert lokaal.
# -------------------------------------------------------------------
@st.cache_data(ttl=86400) 
def fetch_historical_benchmark_data(start_date_str, end_date_str):
    """
    Haalt de historische data van één geselecteerde klimaatnormaalperiode op.
    Roept de complete dataset op en filtert de data lokaal om rate limits te vermijden.
    """
    # Stap 1: Haal de complete dataset op (deze is gecached)
    df_complete, status = fetch_complete_historical_data(BENCHMARK_START_DATE_FULL, BENCHMARK_END_DATE_FULL)
    
    if df_complete.empty:
        return pd.DataFrame(), status

    # Stap 2: Filter de complete data op de gevraagde periode
    df_hist = df_complete[
        (df_complete['Date'] >= start_date_str) & 
        (df_complete['Date'] <= end_date_str)
    ].copy()
    
    if df_hist.empty:
        error_message = f"❌ Geen data gevonden in de complete set voor de periode {start_date_str} tot {end_date_str}."
        return pd.DataFrame(), ('error', error_message)
    
    # Stap 3: Voltooi de verwerking voor de enkelvoudige benchmark
    df_hist['Station Naam'] = 'Historische Benchmark'
    success_message = f"Langjarige benchmarkdata (Klimaatnormaal {start_date_str[:4]}-{end_date_str[:4]}) van {len(df_hist)} dagen succesvol gefilterd van de complete set."
    return df_hist.set_index('Date'), ('success', success_message)

# -------------------------------------------------------------------
# GECORRIGEERDE FUNCTIE: Ophalen Externe Historische Data voor ALLE Klimaatnormalen
# Roept de complete set op en filtert lokaal.
# -------------------------------------------------------------------
@st.cache_data(ttl=86400) # Cache voor 24 uur
def fetch_all_historical_benchmarks(climate_normal_periods):
    """
    Haalt de historische data voor alle gedefinieerde klimaatnormaalperioden op.
    Gebruikt de complete set en filtert lokaal om rate limits te vermijden.
    """
    all_benchmarks = {}
    
    # Stap 1: Haal de complete dataset op (deze is gecached)
    df_complete, status = fetch_complete_historical_data(BENCHMARK_START_DATE_FULL, BENCHMARK_END_DATE_FULL)
    
    if df_complete.empty:
        # Als de complete set niet geladen kan worden, stoppen we hier.
        return all_benchmarks 

    # Sorteer de perioden van oud naar nieuw (laagste startjaar eerst)
    sorted_periods = sorted(climate_normal_periods.items(), key=lambda item: int(item[1][0][:4]), reverse=False)
    
    for period_name, (start_date_str, end_date_str) in sorted_periods:
        
        # Stap 2: Filter de complete data op de gevraagde periode
        df_hist = df_complete[
            (df_complete['Date'] >= start_date_str) & 
            (df_complete['Date'] <= end_date_str)
        ].copy()
        
        if not df_hist.empty:
            df_hist['Periode'] = period_name.split('(')[0].strip() # '1990-2019'
            period_key = period_name.split('(')[0].strip() 
            all_benchmarks[period_key] = df_hist
            
    return all_benchmarks


# Nieuwe, veilige formatteer functie
def safe_format_temp(x):
    """Formats numeric value x to 'X.X °C', returns empty string for NaN or non-numeric types."""
    if pd.isna(x):
        return ""
    try:
        # Probeer om te zetten naar float en formatteer
        return f"{float(x):.1f} °C"
    except (ValueError, TypeError):
        # Vang op als x een onverwacht type is dat niet naar float kan
        return "" 
        
# NIEUWE FUNCTIE: Haalt de eenheid uit de display-naam
def get_unit_from_display_name(display_name, plot_col):
    """
    Haalt de eenheid uit de display naam voor formatting.
    """
    if plot_col in ['temp', 'dauwpunt', 'natbol']:
        return "°C" 
    unit_match = re.search(r'\((.*?)\)', display_name)
    if unit_match:
        return unit_match.group(1).strip()
    return display_name.split(' ')[-1].strip()


def find_consecutive_periods(df_filtered, min_days, temp_column):
    """Vindt en retourneert aaneengesloten periodes in een gefilterde DataFrame."""
    
    if df_filtered.empty:
        return pd.DataFrame(), 0

    df_groups = df_filtered.copy()
    
    if not isinstance(df_groups.index, pd.DatetimeIndex):
        return pd.DataFrame(), 0 
    
    df_groups = df_groups.reset_index().sort_values('Date') 

    grouped = df_groups.groupby('Station Naam')
    all_periods = []

    for name, group in grouped:
        group['new_period'] = (group['Date'].diff().dt.days.fillna(0) > 1).astype(int) 
        group['group_id'] = group['new_period'].cumsum()
        
        periods = group.groupby('group_id').agg(
            StartDatum=('Date', 'min'),
            EindDatum=('Date', 'max'),
            Duur=('Date', 'size'),
            Gemiddelde_Temp_Periode=(temp_column, 'mean')
        ).reset_index(drop=True)
        
        periods['Station Naam'] = name
        
        periods = periods[periods['Duur'] >= min_days]
        all_periods.append(periods)

    if not all_periods:
        return pd.DataFrame(), 0

    periods_combined = pd.concat(all_periods, ignore_index=True)
    
    periods_combined['StartDatum'] = periods_combined['StartDatum'].dt.strftime('%d-%m-%Y')
    periods_combined['EindDatum'] = periods_combined['EindDatum'].dt.strftime('%d-%m-%Y')
    
    periods_combined['Gemiddelde_Temp_Periode'] = periods_combined['Gemiddelde_Temp_Periode'].map(safe_format_temp)

    total_periods = len(periods_combined)
    
    return periods_combined.reset_index(drop=True), total_periods


def find_extreme_days(df_daily_summary, top_n=5):
    """
    Vindt de warmste, koudste en meest extreme dagen uit de dagelijkse samenvatting.
    """
    if df_daily_summary.empty:
        return {}

    df_analysis = df_daily_summary.copy()
    df_analysis['Temp_Range_C'] = df_analysis['Temp_High_C'] - df_analysis['Temp_Low_C']

    results = {}

    extremes_config = {
        'hoogste_max_temp': ('Temp_High_C', False, 'Max Temp (°C)'), 
        'hoogste_min_temp': ('Temp_Low_C', False, 'Min Temp (°C)'),   
        'hoogste_gem_temp': ('Temp_Avg_C', False, 'Gem Temp (°C)'),   
        'laagste_min_temp': ('Temp_Low_C', True, 'Min Temp (°C)'),    
        'laagste_max_temp': ('Temp_High_C', True, 'Max Temp (°C)'),   
        'laagste_gem_temp': ('Temp_Avg_C', True, 'Gem Temp (°C)'),    
        'grootste_range': ('Temp_Range_C', False, 'Range (°C)')
    }

    for key, (column, ascending, display_col) in extremes_config.items():
        
        extreme_df_list = []
        for station, group in df_analysis.groupby('Station Naam'):
            
            top_days = group.sort_values(by=column, ascending=ascending).head(top_n)

            df_display = top_days.reset_index().copy()

            if 'Date' in df_display.columns:
                 df_display['Datum'] = df_display['Date'].dt.strftime('%d-%m-%Y')
                 df_display = df_display.drop(columns=['Date']) 
            
            rename_dict = {
                'Temp_High_C': 'Max Temp (°C)', 
                'Temp_Low_C': 'Min Temp (°C)',
                'Temp_Avg_C': 'Gem Temp (°C)',
                'Temp_Range_C': 'Range (°C)' 
            }
            
            df_display = df_display.rename(columns=rename_dict)
            
            temp_display_cols = ['Max Temp (°C)', 'Min Temp (°C)', 'Range (°C)', 'Gem Temp (°C)']
            for col_name in temp_display_cols:
                if col_name in df_display.columns:
                     df_display[col_name] = df_display[col_name].map(safe_format_temp)


            if key in ['hoogste_max_temp', 'laagste_max_temp']:
                 final_cols_order = ['Datum', 'Station Naam', 'Max Temp (°C)']
            elif key in ['hoogste_min_temp', 'laagste_min_temp']:
                 final_cols_order = ['Datum', 'Station Naam', 'Min Temp (°C)']
            elif key in ['hoogste_gem_temp', 'laagste_gem_temp']:
                 final_cols_order = ['Datum', 'Station Naam', 'Gem Temp (°C)']
            else: # grootste_range
                 final_cols_order = ['Datum', 'Station Naam', 'Range (°C)', 'Max Temp (°C)', 'Min Temp (°C)']
            
            df_final_display = df_display[[c for c in final_cols_order if c in df_display.columns]].copy()
            
            extreme_df_list.append(df_final_display)

        if extreme_df_list:
             results[key] = pd.concat(extreme_df_list, ignore_index=True).sort_values(by='Station Naam', ascending=True)

    return results

def display_extreme_results_by_station(df_results, title, info_text):
    """Toont extreme dagen, gegroepeerd per station met een duidelijke kop."""
    st.markdown(f"### {title}")
    st.info(info_text)
    
    if df_results is None or df_results.empty:
        st.warning("Geen data gevonden voor deze extreme categorie.")
        return

    for station_name, df_station in df_results.groupby('Station Naam'):
        st.markdown(f"##### 📌 Station: **{station_name}**")
        display_cols = [c for c in df_station.columns if c not in ['Station Naam']] 
        st.dataframe(df_station[display_cols].set_index('Datum'), use_container_width=True)
        st.markdown("---")


# 3. Streamlit Applicatie Hoofdsectie (Streamlit Application Main)

st.set_page_config(
    page_title="Weergrafieken & Datazoeker",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(f"☀️ Weergrafieken ")
st.markdown("Analyseer en visualiseer weerdata van de Malmån stations.")

# CRUCIALE FIX: Initialiseer df_combined VOORDAT de sidebar deze gebruikt
df_combined = pd.DataFrame() 
failed_stations = []

# PlaatsHouders voor de sidebar (Nu ook de benchmark status)
years_info_to_display = [] 
button_action = False 
status_placeholder = None 
info_placeholder_start_year = None
info_placeholder_years = None
info_placeholder_last_check = None
info_placeholder_benchmark = None 
all_hist_benchmarks = {} # NIEUW: Initialisatie van de all_hist_benchmarks variabele

# --- Initialiseer de Session State voor de Benchmark Selectie ---
if 'benchmark_period_select' not in st.session_state:
    st.session_state.benchmark_period_select = list(CLIMATE_NORMAL_PERIODS.keys())[0]
    st.session_state.benchmark_start_date, st.session_state.benchmark_end_date = CLIMATE_NORMAL_PERIODS[st.session_state.benchmark_period_select]
    st.session_state.benchmark_period_display = st.session_state.benchmark_period_select
    
# 💥 Sectie 1: Data Selectie & Visualisatie Opties (Geconsolideerd)
with st.sidebar.expander("🛠️ Data & Visualisatie Keuze", expanded=True):
    st.header("1. Stationsselectie")

    station_options = list(STATION_MAP.keys())
    selected_station_ids = st.multiselect(
        "Kies één of meer weerstations:",
        options=station_options,
        default=[station_options[0]] if station_options else [], 
        format_func=lambda x: STATION_MAP[x]
    )

    # --- Scheiding ---
    st.markdown("---")
    st.header("2. Grafiek & Tijd")
    
    # a. Grafiek Variabele
    st.markdown("##### Variabele")
    plot_options = [COL_DISPLAY_MAP[col] for col in NUMERIC_COLS if col in df_combined.columns or df_combined.empty]
    
    if df_combined.empty:
        plot_options = [COL_DISPLAY_MAP[col] for col in NUMERIC_COLS]
        
    default_selection = [COL_DISPLAY_MAP.get('temp')] if COL_DISPLAY_MAP.get('temp') in plot_options else [plot_options[0]] if plot_options else []
         
    selected_variable_display = st.selectbox(
        "Kies de variabele voor de grafiek:",
        options=plot_options,
        index=plot_options.index(default_selection[0]) if default_selection else 0, 
        key='variable_select',
    )
    
    if plot_options:
         selected_variables = [DISPLAY_TO_COL_MAP[selected_variable_display]]
    else:
         selected_variables = []
    
    # b. Tijdsbereik
    st.markdown("##### Tijdsbereik (Grafiek & Ruwe Data)")
    time_range_options = [
        "Huidige dag (sinds 00:00 uur)",
        "Laatste 24 uur",
        "Huidige maand",
        "Huidig jaar",
        "Vrije selectie op datum"
    ]

    selected_range_option = st.selectbox(
        "Kies een tijdsbereik:",
        options=time_range_options,
        index=0,
        key='time_range_select'
    )
    
    # c. Datapunten
    show_markers = st.checkbox(
        "Toon Datapunten (Markers)", 
        key="show_markers",
    )
    

# 💥 Sectie 2: Historische Benchmark Instellingen (NIEUW)
with st.sidebar.expander("🌍 Historische Benchmark Instellingen", expanded=True):
    st.header("Langjarige Normaal")
    
    # Gebruik de eerder gedefinieerde constante
    selected_period_key = st.selectbox(
        "Kies de Klimaat Normaalperiode:",
        options=list(CLIMATE_NORMAL_PERIODS.keys()),
        index=list(CLIMATE_NORMAL_PERIODS.keys()).index(st.session_state.benchmark_period_select), 
        key='benchmark_period_select_input' # Nieuwe key voor de widget
    )
    
    # Update de Session State op basis van de selectie
    st.session_state.benchmark_period_select = selected_period_key
    st.session_state.benchmark_start_date, st.session_state.benchmark_end_date = CLIMATE_NORMAL_PERIODS[selected_period_key]
    st.session_state.benchmark_period_display = selected_period_key.split('(')[0].strip() # Voor nette weergave (bv. '1990-2019')

# 💥 Sectie 3: Programma Checks (Status en Herlaadknop)
with st.sidebar.expander("⚙️ Programma Checks", expanded=False):
    st.header("Systeem Status & Controles")
    
    status_placeholder = st.empty() 

    if st.button("Herlaad Data (Wis Cache)", key="reload_button_check"):
        st.cache_data.clear()
        button_action = True 
    
    st.markdown("---")
    
    info_placeholder_start_year = st.empty()
    info_placeholder_years = st.empty()
    info_placeholder_last_check = st.empty()
    
    # Plaats de Benchmark Status HIER
    info_placeholder_benchmark = st.empty()


# --- Data Loading Logic (Gebruikt de placeholders van hierboven) ---

if button_action:
    st.rerun() 
    
if selected_station_ids:
    
    # 1. Jaarontdekking
    if status_placeholder:
        status_placeholder.info("Zoekt naar beschikbare datajaren op GitHub...", icon="⏳")
        
    available_years = discover_available_years(START_YEAR, selected_station_ids[0], GITHUB_BASE_URL)
    
    if not available_years:
        if status_placeholder:
            status_placeholder.error(f"Geen data gevonden van {START_YEAR} tot nu voor het pad {selected_station_ids[0]}. Controleer URL en startjaar.")
    else:
        years_info_to_display = available_years
        
        all_data = []
        for station_id in selected_station_ids:
            station_name = STATION_MAP.get(station_id, station_id)
            
            # 2. Data Laden
            if status_placeholder:
                status_placeholder.info(f"Data van: **{station_name}** ({', '.join(map(str, available_years))}) wordt geladen...", icon="⬇️")
            
            df_station = load_data(station_id, available_years, GITHUB_BASE_URL, STATION_MAP, TARGET_TIMEZONE)
            
            if not df_station.empty:
                df_station['Station Naam'] = station_name
                all_data.append(df_station)
            else:
                failed_stations.append(station_name)
        
        # 3. Eindstatus
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            
            if failed_stations:
                if status_placeholder:
                    status_placeholder.warning(f"Laden voltooid. Fout bij stations: {', '.join(failed_stations)}")
            else:
                if status_placeholder:
                    status_placeholder.success("Data laden voltooid!", icon="✅")
                
        else:
            if status_placeholder:
                status_placeholder.error(f"Geen data geladen voor de geselecteerde stations uit de gevonden jaren.")

    # 4. Update Programma Info
    if info_placeholder_start_year:
        info_placeholder_start_year.markdown(f"**Start Jaar Zoektocht:** `{START_YEAR}`")
    if info_placeholder_years:
        info_placeholder_years.markdown(f"**Gevonden Jaarbestanden:** `{', '.join(map(str, years_info_to_display))}`")
    if info_placeholder_last_check:
        info_placeholder_last_check.markdown(f"**Laatste Data Check:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    if status_placeholder:
        status_placeholder.info("Selecteer stations om data te laden.", icon="ℹ️")
    if info_placeholder_start_year:
        info_placeholder_start_year.markdown(f"**Start Jaar Zoektocht:** `{START_YEAR}`")
    if info_placeholder_years:
        info_placeholder_years.markdown(f"**Gevonden Jaarbestanden:** Geen. Selecteer stations.")
    if info_placeholder_last_check:
        info_placeholder_last_check.markdown(f"**Laatste Data Check:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# --- Data Verwerking Logica ---

if df_combined.empty:
    pass # Ga verder zonder data indien nodig
else:
    
    # 1. Tijdsfiltering voor Grafiek & Ruwe Data
    now_local = pd.Timestamp.now(tz=TARGET_TIMEZONE)
    min_date_available = df_combined['Timestamp_Local'].min().date()
    max_date_available = df_combined['Timestamp_Local'].max().date()
        
    start_date_local = None
    end_date_local = now_local 
    
    if st.session_state.time_range_select == "Huidige dag (sinds 00:00 uur)":
        start_date_local = now_local.normalize()
    
    elif st.session_state.time_range_select == "Laatste 24 uur":
        start_date_local = now_local - pd.Timedelta(hours=24)
        
    elif st.session_state.time_range_select == "Huidige maand":
        start_date_local = now_local.to_period('M').start_time.tz_localize(TARGET_TIMEZONE)
        
    elif st.session_state.time_range_select == "Huidig jaar":
        start_date_local = now_local.to_period('Y').start_time.tz_localize(TARGET_TIMEZONE)
        
    elif st.session_state.time_range_select == "Vrije selectie op datum":
        if 'custom_date_selector_tab12' not in st.session_state:
            st.session_state.custom_date_selector_tab12 = [max_date_available, max_date_available]
            
        custom_date_range = st.sidebar.date_input(
            "Kies start- en einddatum:",
            value=st.session_state.custom_date_selector_tab12, 
            min_value=min_date_available,
            max_value=max_date_available,
            key='custom_date_selector_tab12_sidebar',
            label_visibility='collapsed'
        )
        st.session_state.custom_date_selector_tab12 = custom_date_range
        
        if len(custom_date_range) == 2:
            start_date_local = pd.to_datetime(custom_date_range[0]).tz_localize(TARGET_TIMEZONE).normalize()
            end_date_local = pd.to_datetime(custom_date_range[1]).tz_localize(TARGET_TIMEZONE).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
        elif len(custom_date_range) == 1:
            start_date_local = pd.to_datetime(custom_date_range[0]).tz_localize(TARGET_TIMEZONE).normalize()
            end_date_local = start_date_local + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            start_date_local = df_combined['Timestamp_Local'].min()
            end_date_local = df_combined['Timestamp_Local'].max()

    if start_date_local is not None and end_date_local is not None:
        
        if start_date_local > end_date_local:
             start_date_local, end_date_local = end_date_local, start_date_local
             
        filtered_df = df_combined[
            (df_combined['Timestamp_Local'] >= start_date_local) & 
            (df_combined['Timestamp_Local'] <= end_date_local)
        ]
        
        if not filtered_df.empty:
            start_display = filtered_df['Timestamp_Local'].min()
            end_display = filtered_df['Timestamp_Local'].max()
        else:
            start_display = start_date_local
            end_display = end_date_local
            
        date_range_display_start = start_display.strftime('%d-%m-%Y %H:%M')
        date_range_display_end = end_display.strftime('%d-%m-%Y %H:%M')

    else:
        filtered_df = df_combined
        date_range_display_start = df_combined['Timestamp_Local'].min().strftime('%d-%m-%Y %H:%M')
        date_range_display_end = df_combined['Timestamp_Local'].max().strftime('%d-%m-%Y %H:%M')
        
    # 2. Dagelijkse Samenvatting (voor Tab 3, 4, 5)
    df_combined_indexed = df_combined.set_index('Timestamp_Local')

    df_daily_summary = df_combined_indexed.groupby('Station Naam').resample('D').agg(
        Temp_High_C=('temp', 'max'),
        Temp_Low_C=('temp', 'min'),
        Temp_Avg_C=('temp', 'mean'),
        Pres_Avg_hPa=('druk', 'mean'), 
        Hum_Avg_P=('luchtvocht', 'mean'), 
    ).dropna(subset=['Temp_Avg_C']).reset_index()

    df_daily_summary = df_daily_summary.rename(columns={'Timestamp_Local': 'Date'}).set_index('Date')
    df_daily_summary.index.name = 'Date' 
    
    # -------------------------------------------------------------
    # Langjarig Gemiddelde (Historical Benchmark) Berekenen
    # -------------------------------------------------------------

    df_hist_raw, benchmark_status = fetch_historical_benchmark_data(
        st.session_state.benchmark_start_date, 
        st.session_state.benchmark_end_date
    ) 
    
    if info_placeholder_benchmark and benchmark_status:
        status_type, message = benchmark_status
        if status_type == 'success':
            info_placeholder_benchmark.success(message, icon="📈")
        elif status_type == 'warning':
            info_placeholder_benchmark.warning(message, icon="⚠️")
        elif status_type == 'error':
            info_placeholder_benchmark.error(message, icon="❌")

    if not df_hist_raw.empty and all(col in df_hist_raw.columns for col in ['Temp_Avg_C', 'Temp_High_C', 'Temp_Low_C']):
        
        df_hist_raw = df_hist_raw.reset_index() 
        df_hist_raw['Month_Day'] = df_hist_raw['Date'].dt.strftime('%m-%d')
        
        # 1. Bereken de Langjarige Benchmark
        df_langjarig_gemiddelde = df_hist_raw.groupby('Month_Day').agg(
            Langjarig_Avg_Temp=('Temp_Avg_C', 'mean'),
            Langjarig_Avg_Max=('Temp_High_C', 'mean'),
            Langjarig_Avg_Min=('Temp_Low_C', 'mean'),
        ).reset_index()
        
        df_langjarig_gemiddelde = df_langjarig_gemiddelde.set_index('Month_Day')
        
        # 2. Voeg de benchmark toe aan df_daily_summary
        df_daily_summary_extended = df_daily_summary.copy().reset_index()
        df_daily_summary_extended['Month_Day'] = df_daily_summary_extended['Date'].dt.strftime('%m-%d')
        
        df_daily_summary = pd.merge(
            df_daily_summary_extended,
            df_langjarig_gemiddelde,
            on='Month_Day',
            how='left'
        ).drop(columns=['Month_Day']).set_index('Date')
        
        df_daily_summary.index.name = 'Date' 
    
    # -------------------------------------------------------------
    # NIEUW: Ophalen van ALLE Historische Benchmarks voor Klimatologie Tab
    # -------------------------------------------------------------
    all_hist_benchmarks = fetch_all_historical_benchmarks(CLIMATE_NORMAL_PERIODS)


# -------------------------------------------------------------------
# --- Hoofdsectie met St.Tabs (Verbeterde Navigatie) ---
# -------------------------------------------------------------------

if df_combined.empty:
    st.warning("Geen weerdata geladen. Selecteer stations in de zijbalk.")
else:
    
    tab_titles_full = ["📈 Grafiek", "📊 Ruwe Data", "🔍 Historische Zoeker", "⭐ Maand/Jaar Analyse", "🏆 Extremen", "🌎 Klimatologie"] # <--- HIER IS DE WIJZIGING
    
    tab_graph, tab_raw, tab_history, tab_analysis, tab_extremes, tab_climatology = st.tabs(tab_titles_full) # <--- HIER IS DE WIJZIGING
    
    # --- Tab 1: Grafiek ---
    with tab_graph:
        
        st.header("Grafiek Weergave")
        st.markdown(f"**Periode:** {date_range_display_start} tot {date_range_display_end} (Tijdzone: {TARGET_TIMEZONE})")
        st.markdown(f"**Gevisualiseerde Variabele:** **{selected_variable_display}**")
        
        if not filtered_df.empty:
            
            # --- Samenvattingssectie: Kernwaarden (st.metric) ---
            st.markdown("---")
            st.subheader("📊 Kernwaarden van de Geselecteerde Periode")
            
            plot_col = selected_variables[0] 
            y_axis_title = selected_variable_display
            unit = get_unit_from_display_name(y_axis_title, plot_col)

            def format_value_metric(val):
                if pd.isna(val):
                    return "N/A"
                return f"{val:.1f} {unit}"
            
            for station_name, group in filtered_df.groupby('Station Naam'):
                
                st.markdown(f"##### 📌 Station: **{station_name}**")

                max_val = group[plot_col].max()
                avg_val = group[plot_col].mean()
                min_val = group[plot_col].min()

                last_row = group.sort_values('Timestamp_Local', ascending=False).iloc[0]
                last_val = last_row[plot_col]
                last_time = last_row['Timestamp_Local'].strftime('%H:%M') 
                
                last_delta_val_str = None
                if not pd.isna(last_val) and not pd.isna(avg_val):
                     delta_raw = last_val - avg_val
                     if delta_raw > 0:
                         last_delta_val_str = f"+{delta_raw:.1f} {unit}"
                     elif delta_raw < 0:
                          last_delta_val_str = f"{delta_raw:.1f} {unit}"
                     else:
                          last_delta_val_str = "0.0 " + unit
                
                col_last, col_min, col_avg, col_max = st.columns(4)

                with col_last:
                    st.metric(
                        label=f"Laatste ({last_time})", 
                        value=format_value_metric(last_val),
                        delta=f"vs. Gem: {last_delta_val_str}" if last_delta_val_str else None,
                        delta_color="normal"
                    )
                with col_min:
                    min_row = group[group[plot_col] == min_val].sort_values('Timestamp_Local', ascending=False).iloc[0]
                    min_time = min_row['Timestamp_Local'].strftime('%H:%M')
                    st.metric(label=f"Minimale ({min_time})", value=format_value_metric(min_val))
                with col_avg:
                    st.metric(label="Gemiddelde", value=format_value_metric(avg_val))
                with col_max:
                    max_row = group[group[plot_col] == max_val].sort_values('Timestamp_Local', ascending=False).iloc[0]
                    max_time = max_row['Timestamp_Local'].strftime('%H:%M')
                    st.metric(label=f"Maximale ({max_time})", value=format_value_metric(max_val))
                st.markdown("---") 
            
            st.markdown("---") 
            
            # --- Grafiek ---
            plot_col = selected_variables[0] 
            y_axis_title = selected_variable_display
            
            trace_mode = 'lines+markers' if st.session_state.get('show_markers') else 'lines'

            fig = px.line(
                filtered_df,
                x='Timestamp_Local',
                y=plot_col,
                color='Station Naam',
                line_shape='linear',
                title=f'Weerdata van {selected_variable_display} ({date_range_display_start} - {date_range_display_end})',
                labels={'Timestamp_Local': f'Tijdstip ({TARGET_TIMEZONE})', plot_col: y_axis_title},
                height=600
            )

            trace_hovertemplate_tab1 = (
                f"<b>{y_axis_title}:</b> %{{y:.1f}} {unit}<br>" +
                "<b>Station:</b> %{full_data.name}<extra></extra>" 
            )
            
            fig.update_traces(
                mode=trace_mode, 
                hovertemplate=trace_hovertemplate_tab1
            ) 
            
            fig.update_layout(
                hovermode="x unified",
                xaxis_title=f'Tijdstip ({TARGET_TIMEZONE})',
                yaxis_title=y_axis_title,
                legend_title_text='Station',
            )

            fig.update_xaxes(
                 title=f'Tijdstip ({TARGET_TIMEZONE})',
                 hoverformat="%d-%m-%Y %H:%M",
                 tickformat="%H:%M\n%d-%m" 
            )

            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Geen data gevonden voor het geselecteerde tijdsbereik en station(s).")

    # --- Tab 2: Ruwe Data ---
    with tab_raw:
        
        st.header("Ruwe Data")
        st.markdown(f"**Periode:** {date_range_display_start} tot {date_range_display_end} (Tijdzone: {TARGET_TIMEZONE})")
        st.info(f"Dit zijn de ruwe meetwaarden. Getoond zijn de kolommen: `Timestamp_Local`, `Station Naam`, en de gekozen variabelen: `{', '.join(selected_variables)}`")

        if not filtered_df.empty:
            cols_to_display = ['Timestamp_Local', 'Station Naam'] + selected_variables
            df_display_raw = filtered_df[cols_to_display].sort_values('Timestamp_Local', ascending=False)
            
            rename_dict_raw = {col: COL_DISPLAY_MAP.get(col, col) for col in cols_to_display}
            df_display_raw = df_display_raw.rename(columns=rename_dict_raw)

            st.dataframe(df_display_raw, use_container_width=True, height=700)
            
        else:
            st.warning("Geen data gevonden voor het geselecteerde tijdsbereik en station(s).")


    # --- Tab 3: Historische Zoeker ---
    with tab_history:
        
        st.header("Historische Zoeker (Dagelijkse Samenvatting)")
        
        if df_daily_summary.empty:
            st.warning("Geen dagelijkse samenvatting beschikbaar. Laad eerst data.")
        else:
            
            with st.expander("⚙️ Pas Historische Zoekfilters aan", expanded=True):
                
                st.markdown(f"**1. Zoekperiode**")
                period_options = ["Onbeperkt (Volledige Database)", "Selecteer Jaar", "Selecteer Maand", "Aangepaste Datums"]
                period_type = st.selectbox(
                    "1. Zoekperiode (Historisch)", 
                    period_options, 
                    key="period_select_hist"
                )
                
                df_filtered_time = df_daily_summary.copy()
                
                min_date_hist = df_daily_summary.index.min().date()
                max_date_hist = df_daily_summary.index.max().date()
                
                if period_type == "Selecteer Jaar":
                    available_years_hist = sorted(df_daily_summary.index.year.unique())
                    st.selectbox(
                        "Kies Jaar:", 
                        available_years_hist, 
                        key="hist_year"
                    )
                    df_filtered_time = df_daily_summary[df_daily_summary.index.year == st.session_state.hist_year]
                
                elif period_type == "Selecteer Maand":
                    df_temp_filter = df_daily_summary.copy()
                    df_temp_filter['JaarMaand'] = df_temp_filter.index.to_period('M').astype(str)
                    available_periods = sorted(df_temp_filter['JaarMaand'].unique(), reverse=True)
                    
                    st.selectbox(
                        "Kies Jaar en Maand (YYYY-MM):", 
                        available_periods, 
                        index=0, 
                        key="hist_year_month"
                    )
                    selected_period_str_hist = st.session_state.hist_year_month
                    df_filtered_time = df_temp_filter[df_temp_filter['JaarMaand'] == selected_period_str_hist].drop(columns=['JaarMaand'])

                elif period_type == "Aangepaste Datums":
                    if 'hist_dates' not in st.session_state:
                         st.session_state.hist_dates = (min_date_hist, max_date_hist)
                         
                    st.date_input(
                        "Kies Start- en Einddatum:",
                        value=st.session_state.hist_dates,
                        min_value=min_date_hist,
                        max_value=max_date_hist,
                        key="hist_dates_input"
                    )
                    st.session_state.hist_dates = st.session_state.hist_dates_input 
                    
                    date_range_hist = st.session_state.hist_dates
                    
                    if len(date_range_hist) == 2:
                        df_filtered_time = df_daily_summary.loc[str(date_range_hist[0]):str(date_range_hist[1])]
                    elif len(date_range_hist) == 1:
                        df_filtered_time = df_daily_summary.loc[str(date_range_hist[0]):str(date_range_hist[0])]


                st.markdown("---")
                st.markdown(f"**2. Filtertype & Drempel**")

                filter_mode = st.radio(
                    "Zoeken op:",
                    ["Losse Dagen", "Aaneengesloten Periode", "Hellmann Getal Berekenen"],
                    key="filter_mode"
                )

                temp_column = 'Temp_Avg_C'
                comparison = "Lager dan (<=)" 
                temp_threshold = 0.0
                min_consecutive_days = 0

                if filter_mode == "Hellmann Getal Berekenen":
                    st.markdown("---")
                    st.markdown(f"**3. Hellmann Berekening**")
                    st.markdown("**Basis:** Absolute som van Gemiddelde Dagtemp $\\le 0.0$ °C.")
                    
                elif filter_mode == "Aaneengesloten Periode":
                    st.markdown("---")
                    st.markdown(f"**3. Periodedrempel**")
                    min_consecutive_days = st.number_input(
                        "Min. aaneengesloten dagen:",
                        min_value=2,
                        value=3,
                        step=1,
                        key="min_days_period"
                    )
                    st.markdown("---")
                    st.markdown(f"**4. Temperatuurfilter**")
                    temp_type_options = ["Max Temp (Temp_High_C)", "Min Temp (Temp_Low_C)", "Gemiddelde Temp (Temp_Avg_C)"]
                    temp_type = st.selectbox(
                        "Meetwaarde:", 
                        temp_type_options, 
                        index=2, 
                        key="temp_type_period"
                    )
                    temp_column = temp_type.split(" (")[1][:-1] if " (" in temp_type else 'Temp_Avg_C'
                    
                    comparison = st.radio(
                        "Vergelijking:", 
                        ["Hoger dan (>=)", "Lager dan (<=)"], 
                        key="comparison_period"
                    ) 
                    
                    temp_threshold = st.number_input(
                        "Temperatuur (°C):", 
                        value=15.0, 
                        step=0.5, 
                        key="temp_threshold_period"
                    )

                else: # Losse Dagen
                    st.markdown("---")
                    st.markdown(f"**4. Temperatuurfilter**")
                    temp_type_options = ["Max Temp (Temp_High_C)", "Min Temp (Temp_Low_C)", "Gemiddelde Temp (Temp_Avg_C)"]
                    temp_type = st.selectbox(
                        "Meetwaarde:", 
                        temp_type_options, 
                        index=2, 
                        key="temp_type_days"
                    )
                    temp_column = temp_type.split(" (")[1][:-1] if " (" in temp_type else 'Temp_Avg_C'
                    
                    comparison = st.radio(
                        "Vergelijking:", 
                        ["Hoger dan (>=)", "Lager dan (<=)"], 
                        key="comparison_days"
                    )
                    
                    temp_threshold = st.number_input(
                        "Temperatuur (°C):", 
                        value=15.0, 
                        step=0.5, 
                        key="temp_threshold_days"
                    )
            
            filtered_data = df_filtered_time.copy()

            if filter_mode == "Hellmann Getal Berekenen":
                
                st.subheader(f"❄️ Hellmann Getal Berekening")
                start_date = df_filtered_time.index.min().strftime('%d-%m-%Y')
                end_date = df_filtered_time.index.max().strftime('%d-%m-%Y')
                
                st.info(f"**Analyse Periode:** {period_type} van **{start_date}** tot **{end_date}**.")

                df_hellmann_days = filtered_data[filtered_data['Temp_Avg_C'] <= 0.0].copy()
                
                if df_hellmann_days.empty:
                    st.success("Er zijn geen dagen gevonden met een Gemiddelde Dagtemperatuur van 0.0 °C of lager in deze periode.")
                else:
                    hellmann_results = df_hellmann_days.groupby('Station Naam').agg(
                        Aantal_Dagen_Koude=('Temp_Avg_C', 'size'),
                        Hellmann_Getal=('Temp_Avg_C', lambda x: x[x <= 0].abs().sum().round(1))
                    ).reset_index()
                    
                    hellmann_results['Hellmann Getal'] = hellmann_results['Hellmann_Getal'].astype(str) + " °C"
                    hellmann_results = hellmann_results.rename(columns={
                        'Aantal_Dagen_Koude': 'Aantal Dagen (≤ 0.0 °C)',
                    }).set_index('Station Naam').drop(columns=['Hellmann_Getal'])
                    
                    hellmann_results = hellmann_results[['Hellmann Getal', 'Aantal Dagen (≤ 0.0 °C)']]
                    
                    st.success("Hellmann Getal is de absolute som van de Gemiddelde Dagtemperatuur (alleen als de temperatuur $\\le 0.0$ °C is).")
                    st.dataframe(hellmann_results, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Gevonden Dagen ($\le 0.0$ °C Gem. Temp)")
                    
                    df_hellmann_days_display = df_hellmann_days[['Station Naam', 'Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C']].reset_index()
                    df_hellmann_days_display = df_hellmann_days_display.rename(columns={'Date': 'Datum', 'Temp_High_C': 'Max Temp', 'Temp_Low_C': 'Min Temp', 'Temp_Avg_C': 'Gem Temp'})
                    
                    for col in ['Max Temp', 'Min Temp', 'Gem Temp']:
                        df_hellmann_days_display[col] = df_hellmann_days_display[col].map(safe_format_temp)
                        
                    df_hellmann_days_display = df_hellmann_days_display.sort_values(['Station Naam', 'Datum'], ascending=[True, False]).set_index('Datum')
                    
                    for station, df_group in df_hellmann_days_display.groupby('Station Naam'):
                        st.markdown(f"##### 📌 Station: **{station}** ({len(df_group)} dagen)")
                        st.dataframe(df_group.drop(columns=['Station Naam']), use_container_width=True)
                        st.markdown("---")


            elif filter_mode == "Aaneengesloten Periode":
                
                display_temp_type = st.session_state.get('temp_type_period', 'Gemiddelde Temp (Temp_Avg_C)').split(" (")[0]
                comparison_char = "≥" if comparison == "Hoger dan (>=)" else "≤"
                start_date = df_filtered_time.index.min().strftime('%d-%m-%Y')
                end_date = df_filtered_time.index.max().strftime('%d-%m-%Y')
                
                st.subheader(f"🔥 Zoekresultaten: Aaneengesloten Periodes van **{min_consecutive_days}**+ dagen")
                
                st.info(f"""
                **Criteria:** {min_consecutive_days}+ aaneengesloten dagen waarbij **{display_temp_type}** {comparison_char} **{temp_threshold}°C**
                
                📆 **Geselecteerde Periode:** {start_date} tot {end_date}
                """)

                if comparison == "Hoger dan (>=)":
                    df_filtered_period = filtered_data[filtered_data[temp_column] >= temp_threshold]
                else:
                    df_filtered_period = filtered_data[filtered_data[temp_column] <= temp_threshold]

                periods_df, total_periods = find_consecutive_periods(df_filtered_period, min_consecutive_days, temp_column)
                
                if total_periods > 0:
                    st.success(f"✅ **{total_periods}** aaneengesloten periodes van {min_consecutive_days} dagen of meer gevonden.")
                    
                    periods_df_display = periods_df.rename(columns={'Duur': 'Duur (Dagen)', 'Gemiddelde_Temp_Periode': f'Gem. {temp_column}'})
                    
                    st.dataframe(periods_df_display.set_index('Station Naam'), use_container_width=True)
                    
                else:
                    st.warning("Geen aaneengesloten periodes gevonden die aan de criteria voldoen.")

            else: # Losse Dagen
                
                display_temp_type = st.session_state.get('temp_type_days', 'Gemiddelde Temp (Temp_Avg_C)').split(" (")[0]
                comparison_char = "≥" if comparison == "Hoger dan (>=)" else "≤"
                start_date = df_filtered_time.index.min().strftime('%d-%m-%Y')
                end_date = df_filtered_time.index.max().strftime('%d-%m-%Y')
                
                st.subheader(f"🔍 Zoekresultaten: Losse Dagen")
                
                st.info(f"""
                **Criteria:** Dagen waarbij **{display_temp_type}** {comparison_char} **{temp_threshold}°C**
                
                📆 **Geselecteerde Periode:** {start_date} tot {end_date}
                """)
                
                if comparison == "Hoger dan (>=)":
                    df_filtered_days = filtered_data[filtered_data[temp_column] >= temp_threshold].copy()
                else:
                    df_filtered_days = filtered_data[filtered_data[temp_column] <= temp_threshold].copy()
                
                total_days = len(df_filtered_days)
                
                if total_days > 0:
                    st.success(f"✅ **{total_days}** dagen gevonden die aan de criteria voldoen.")
                    
                    df_filtered_days_display = df_filtered_days[['Station Naam', 'Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C']].reset_index()
                    df_filtered_days_display = df_filtered_days_display.rename(columns={'Date': 'Datum', 'Temp_High_C': 'Max Temp', 'Temp_Low_C': 'Min Temp', 'Temp_Avg_C': 'Gem Temp'})
                    
                    for col in ['Max Temp', 'Min Temp', 'Gem Temp']:
                        df_filtered_days_display[col] = df_filtered_days_display[col].map(safe_format_temp)
                        
                    df_filtered_days_display = df_filtered_days_display.sort_values(['Station Naam', 'Datum'], ascending=[True, False]).set_index('Datum')
                    
                    for station, df_group in df_filtered_days_display.groupby('Station Naam'):
                        st.markdown(f"##### 📌 Station: **{station}** ({len(df_group)} dagen)")
                        st.dataframe(df_group.drop(columns=['Station Naam']), use_container_width=True)
                        st.markdown("---")
                else:
                    st.warning("Geen dagen gevonden die aan de criteria voldoen in deze periode.")


    # --- Tab 4: Maand/Jaar Analyse ---
    with tab_analysis:
        
        st.header(f"Maand/Jaar Analyse")
        
        if df_daily_summary.empty:
            st.warning("Geen dagelijkse data gevonden. Laad eerst data.")
        else:
            
            with st.expander("⚙️ Kies Analyse Periode", expanded=True):

                analysis_type = st.radio(
                    "Kies Analyse Niveau:",
                    ["Dag", "Maand", "Jaar"], 
                    key="analysis_level_new"
                )

                df_analysis_selector = df_daily_summary.copy()

                min_date_hist = df_daily_summary.index.min().date()
                max_date_hist = df_daily_summary.index.max().date()

                selected_period_str_analysis = None

                if analysis_type == "Dag":
                    
                    default_date = max_date_hist
                         
                    selected_date = st.date_input(
                        "Kies de Analyse Datum:",
                        value=default_date,
                        min_value=min_date_hist,
                        max_value=max_date_hist,
                        key="analysis_day_select"
                    )
                    
                    df_analysis_selector = df_analysis_selector[
                        df_analysis_selector.index.date == selected_date
                    ]
                    selected_period_str_analysis = selected_date.strftime('%d-%m-%Y')

                elif analysis_type == "Maand":
                    df_analysis_selector['JaarMaand'] = df_analysis_selector.index.to_period('M').astype(str)
                    available_periods = sorted(df_analysis_selector['JaarMaand'].unique(), reverse=True)
                    titel_periode = "Jaar/Maand"

                    selected_period_str_analysis = st.selectbox(
                        f"Kies de Analyse {titel_periode}:",
                        available_periods,
                        index=0,
                        key="analysis_period_select_new_month"
                    )

                    df_analysis_selector = df_analysis_selector[df_analysis_selector['JaarMaand'] == selected_period_str_analysis]

                else: # Jaar
                    df_analysis_selector['Jaar'] = df_analysis_selector.index.year
                    available_periods = sorted(df_analysis_selector['Jaar'].unique(), reverse=True)
                    titel_periode = "Jaar"

                    selected_period_str_analysis = st.selectbox(
                        f"Kies de Analyse {titel_periode}:",
                        available_periods,
                        index=0,
                        key="analysis_period_select_new_year"
                    )

                    df_analysis_selector = df_analysis_selector[df_analysis_selector['Jaar'] == selected_period_str_analysis]

                df_analysis_selector = df_analysis_selector.drop(columns=[c for c in ['JaarMaand', 'Jaar'] if c in df_analysis_selector.columns], errors='ignore')

            if df_analysis_selector.empty:
                st.warning(f"Geen dagelijkse data beschikbaar voor de geselecteerde periode: **{selected_period_str_analysis}**.")
                st.stop()

            st.subheader(f"Overzicht per Station voor **{selected_period_str_analysis}**")

            # --- Samenvattende Tabel (Wordt nu ook gebruikt voor Daganalyse) ---
            
            if analysis_type == "Dag":
                
                df_summary_stats = df_analysis_selector.reset_index().set_index('Station Naam')
                
                df_summary_stats = df_summary_stats.rename(columns={
                     'Temp_High_C': 'Abs. Max Temp (°C)',
                     'Temp_Low_C': 'Abs. Min Temp (°C)',
                     'Temp_Avg_C': 'Gem. Temp Periode (°C)',
                     'Pres_Avg_hPa': 'Gem. Druk (hPa)',
                     'Hum_Avg_P': 'Gem. Vocht (%)',
                })
                
                df_summary_stats['Aantal Dagen'] = 1 
                
                if 'Langjarig_Avg_Temp' in df_summary_stats.columns:
                     df_summary_stats['Benchmark Gem (°C)'] = df_summary_stats['Langjarig_Avg_Temp']
                     df_summary_stats['Benchmark Max (°C)'] = df_summary_stats['Langjarig_Avg_Max']
                     df_summary_stats['Benchmark Min (°C)'] = df_summary_stats['Langjarig_Avg_Min']
                     
                     col_order_template = [
                        'Aantal Dagen', 
                        'Abs. Max Temp (°C)', 'Benchmark Max (°C)', 
                        'Abs. Min Temp (°C)', 'Benchmark Min (°C)', 
                        'Gem. Temp Periode (°C)', 'Benchmark Gem (°C)', 
                        'Gem. Druk (hPa)', 'Gem. Vocht (%)', 
                     ]
                     
                else:
                    col_order_template = [
                        'Aantal Dagen', 'Abs. Max Temp (°C)', 'Abs. Min Temp (°C)', 
                        'Gem. Temp Periode (°C)', 'Gem. Druk (hPa)', 'Gem. Vocht (%)'
                    ]
                
                df_summary_stats = df_summary_stats[[col for col in col_order_template if col in df_summary_stats.columns]]
                
                for col in [c for c in df_summary_stats.columns if 'Temp' in c or 'Benchmark' in c]:
                    if 'Benchmark' not in col:
                        df_summary_stats[col] = df_summary_stats[col].map(safe_format_temp)
                    else:
                         df_summary_stats[col] = df_summary_stats[col].apply(lambda x: f"{x:.1f} °C")
                         
                if 'Gem. Druk (hPa)' in df_summary_stats.columns:
                    df_summary_stats['Gem. Druk (hPa)'] = df_summary_stats['Gem. Druk (hPa)'].apply(lambda x: f"{x:.1f} hPa")
                if 'Gem. Vocht (%)' in df_summary_stats.columns:
                    df_summary_stats['Gem. Vocht (%)'] = df_summary_stats['Gem. Vocht (%)'].apply(lambda x: f"{x:.1f} %")


            else: # Maand of Jaar Analyse (Originele logica)
                df_summary_stats = df_analysis_selector.groupby('Station Naam').agg(
                    Dagen=('Temp_Avg_C', 'size'),
                    Max_Temp_Abs=('Temp_High_C', 'max'),
                    Min_Temp_Abs=('Temp_Low_C', 'min'),
                    Avg_Temp=('Temp_Avg_C', 'mean'),
                    Avg_Druk=('Pres_Avg_hPa', 'mean'),
                    Avg_Vocht=('Hum_Avg_P', 'mean'),
                )

                if 'Langjarig_Avg_Temp' in df_analysis_selector.columns:

                    df_benchmark_row = df_analysis_selector.reset_index()[['Date', 'Langjarig_Avg_Temp', 'Langjarig_Avg_Max', 'Langjarig_Avg_Min']].drop_duplicates()

                    df_summary_stats['Benchmark Gem. Temp'] = df_benchmark_row['Langjarig_Avg_Temp'].mean()
                    df_summary_stats['Benchmark Max Temp'] = df_benchmark_row['Langjarig_Avg_Max'].mean()
                    df_summary_stats['Benchmark Min Temp'] = df_benchmark_row['Langjarig_Avg_Min'].mean()


                df_summary_stats_display = df_summary_stats.rename(columns={
                    'Dagen': 'Aantal Dagen',
                    'Max_Temp_Abs': 'Abs. Max Temp (°C)',
                    'Min_Temp_Abs': 'Abs. Min Temp (°C)',
                    'Avg_Temp': 'Gem. Temp Periode (°C)',
                    'Avg_Druk': 'Gem. Druk (hPa)',
                    'Avg_Vocht': 'Gem. Vocht (%)',
                    'Benchmark Gem. Temp': 'Benchmark Gem (°C)',
                    'Benchmark Max Temp': 'Benchmark Max (°C)',
                    'Benchmark Min Temp': 'Benchmark Min (°C)',
                })

                df_summary_stats_display['Abs. Max Temp (°C)'] = df_summary_stats_display['Abs. Max Temp (°C)'].map(safe_format_temp)
                df_summary_stats_display['Abs. Min Temp (°C)'] = df_summary_stats_display['Abs. Min Temp (°C)'].map(safe_format_temp)
                df_summary_stats_display['Gem. Temp Periode (°C)'] = df_summary_stats_display['Gem. Temp Periode (°C)'].apply(lambda x: f"{x:.1f} °C")
                df_summary_stats_display['Gem. Druk (hPa)'] = df_summary_stats_display['Gem. Druk (hPa)'].apply(lambda x: f"{x:.1f} hPa")
                df_summary_stats_display['Gem. Vocht (%)'] = df_summary_stats_display['Gem. Vocht (%)'].apply(lambda x: f"{x:.1f} %")

                for col in ['Benchmark Gem (°C)', 'Benchmark Max (°C)', 'Benchmark Min (°C)']:
                     if col in df_summary_stats_display.columns:
                          df_summary_stats_display[col] = df_summary_stats_display[col].apply(lambda x: f"{x:.1f} °C")
                
                df_summary_stats = df_summary_stats_display
                
                col_order_template = [
                    'Aantal Dagen', 
                    'Abs. Max Temp (°C)', 'Benchmark Max (°C)', 
                    'Abs. Min Temp (°C)', 'Benchmark Min (°C)', 
                    'Gem. Temp Periode (°C)', 'Benchmark Gem (°C)', 
                    'Gem. Druk (hPa)', 'Gem. Vocht (%)', 
                ]
                
                final_col_order = [col for col in col_order_template if col in df_summary_stats.columns]
                
                df_summary_stats = df_summary_stats[final_col_order]

            st.dataframe(df_summary_stats, use_container_width=True)
            
            # --- Dagelijkse Lijngrafiek (Alleen voor Maand/Jaar Analyse) ---
            if analysis_type != "Dag":
                st.markdown("---")
                # GEBRUIK DE DYNAMISCHE DISPLAY NAAM
                st.subheader(f"Dagelijkse Min, Max en Gemiddelde Temperatuur vs. Historische Benchmark ({st.session_state.benchmark_period_display})")

                langjarig_cols = ['Langjarig_Avg_Temp', 'Langjarig_Avg_Max', 'Langjarig_Avg_Min']
                existing_langjarig_cols = [col for col in langjarig_cols if col in df_analysis_selector.columns]

                df_plot_combined = pd.DataFrame()

                if not existing_langjarig_cols:
                     df_plot_combined = df_analysis_selector.reset_index().melt(
                         id_vars=['Date', 'Station Naam'],
                         value_vars=['Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C'],
                         var_name='Temperatuur Type',
                         value_name='Temperatuur (°C)'
                     )
                else:
                     df_benchmark_base = df_analysis_selector.reset_index()[['Date'] + existing_langjarig_cols].drop_duplicates()

                     # De benchmark waarden zijn per dag gelijk, maar we moeten ze 'smelten' voor de plot
                     df_benchmark_base = df_benchmark_base.melt(
                         id_vars=['Date'],
                         value_vars=existing_langjarig_cols,
                         var_name='Temperatuur Type',
                         value_name='Temperatuur (°C)'
                     )
                     df_benchmark_base['Station Naam'] = 'Historische Benchmark'

                     df_plot_daily = df_analysis_selector.reset_index().melt(
                         id_vars=['Date', 'Station Naam'],
                         value_vars=['Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C'],
                         var_name='Temperatuur Type',
                         value_name='Temperatuur (°C)'
                     )

                     df_plot_combined = pd.concat([df_plot_daily, df_benchmark_base], ignore_index=True)


                if not df_plot_combined.empty:

                     # Pas de display naam aan voor de benchmark
                     benchmark_period_display = st.session_state.benchmark_period_display.replace(' (Standaard WMO Normaal)', '')
                     
                     temp_map = {
                        'Temp_High_C': 'Max Huidige Periode',
                        'Temp_Low_C': 'Min Huidige Periode',
                        'Temp_Avg_C': 'Gemiddeld Huidige Periode',
                        # Gebruik de dynamische naam in de map
                        'Langjarig_Avg_Temp': f'Benchmark Gemiddeld ({benchmark_period_display})',
                        'Langjarig_Avg_Max': f'Benchmark Max ({benchmark_period_display})',
                        'Langjarig_Avg_Min': f'Benchmark Min ({benchmark_period_display})'
                     }

                     df_plot_combined['Temperatuur Type'] = df_plot_combined['Temperatuur Type'].map(temp_map)

                     line_dash_map = {station_name: 'solid' for station_name in df_plot_combined['Station Naam'].unique()}
                     if 'Historische Benchmark' in line_dash_map:
                         line_dash_map['Historische Benchmark'] = 'dash' 


                     fig_daily = px.line(
                         df_plot_combined,
                         x='Date',
                         y='Temperatuur (°C)',
                         color='Temperatuur Type',
                         line_dash='Station Naam',
                         line_dash_map=line_dash_map,
                         title=f'Dagelijkse Min, Max en Gem. Temp. vs. Historische Benchmark ({benchmark_period_display}) in {selected_period_str_analysis}',
                         labels={'Date': 'Datum', 'Temperatuur (°C)': 'Temperatuur (°C)'},
                         height=600,
                         custom_data=['Station Naam', 'Temperatuur Type']
                     )

                     trace_hovertemplate_tab4 = (
                         "<b>Datum:</b> %{x|%d-%m-%Y}<br>" +
                         "<b>%{customdata[1]}:</b> %{y:.1f} °C<br>" +
                         "<b>Station:</b> %{customdata[0]}<extra></extra>"
                     )

                     fig_daily.update_traces(
                         mode='lines',
                         hovertemplate=trace_hovertemplate_tab4
                     )

                     fig_daily.update_layout(hovermode="x unified")

                     fig_daily.update_xaxes(
                          title='Datum',
                          hoverformat="%d-%m-%Y"
                     )

                     st.plotly_chart(fig_daily, use_container_width=True)
                else:
                     st.warning("Geen data beschikbaar om te plotten in Tab 4.")


    # --- Tab 5: Extremen ---
    with tab_extremes:
        
        st.header("Analyse van Historische Extremen")
        
        if df_daily_summary.empty:
            st.warning("Geen dagelijkse samenvatting beschikbaar. Laad eerst data.")
        else:
            
            extreme_results_full = find_extreme_days(df_daily_summary)

            tab_high_max, tab_low_min, tab_high_min, tab_low_max, tab_high_avg, tab_low_avg, tab_range = st.tabs([
                "Hoogste Max", "Laagste Min", "Warmste Nacht", "Koudste Dag", "Hoogste Gem", "Laagste Gem", "Grootste Range"
            ])

            with tab_high_max:
                display_extreme_results_by_station(
                    extreme_results_full.get('hoogste_max_temp'),
                    "Hoogste Maximum Temperatuur (Top 5 per station) - De Warmste Dagen",
                    "De dagen waarop de temperatuur overdag het hoogst werd gemeten."
                )

            with tab_low_min:
                display_extreme_results_by_station(
                    extreme_results_full.get('laagste_min_temp'), 
                    "Laagste Minimum Temperatuur (Top 5 per station) - De Koudste Nachten",
                    "De nachten/momenten waarop de temperatuur het diepst zakte."
                )
                
            with tab_high_min:
                display_extreme_results_by_station(
                    extreme_results_full.get('hoogste_min_temp'), 
                    "Hoogste Minimum Temperatuur (Top 5 per station) - De Warmste Nachten",
                    "De nachten waarop de temperatuur het hoogst bleef (koelde het minst af)."
                )
            
            with tab_low_max:
                display_extreme_results_by_station(
                    extreme_results_full.get('laagste_max_temp'), 
                    "Laagste Maximum Temperatuur (Top 5 per station) - De Koudste Dagen", 
                    "De dagen waarop de temperatuur overdag het laagst bleef (warmde het minst op)."
                )
            
            with tab_high_avg:
                display_extreme_results_by_station(
                    extreme_results_full.get('hoogste_gem_temp'), 
                    "Hoogste Gemiddelde Temperatuur (Top 5 per station) - De Warmste Gemiddelde Dagen", 
                    "De dagen met het hoogste gemiddelde over 24 uur."
                )

            with tab_low_avg:
                display_extreme_results_by_station(
                    extreme_results_full.get('laagste_gem_temp'), 
                    "Laagste Gemiddelde Temperatuur (Top 5 per station) - De Koudste Gemiddelde Dagen", 
                    "De dagen met het laagste gemiddelde over 24 uur."
                )
                    
            with tab_range:
                display_extreme_results_by_station(
                    extreme_results_full.get('grootste_range'), 
                    "Grootste Dagelijkse Temperatuurbereik (Top 5 Range per station)", 
                    "De Dagelijkse Range is het verschil tussen de Max Temp en de Min Temp op die dag (Grote schommelingen)."
                )


    # --- Tab 6: Klimatologie (NIEUW) ---
    with tab_climatology:
        
        st.header("🌎 Klimatologie: Vergelijking per Periode")
        st.info("Vergelijk de temperatuurextremen en gemiddeldes van een gekozen dag, maand of jaar met alle gedefinieerde historische klimaatnormalen (Langjarige Gemiddeldes van 30 jaar).")
        
        if not all_hist_benchmarks:
            st.error("Geen historische benchmark data beschikbaar. Controleer de Open-Meteo API verbinding in de zijbalk (Programma Checks).")
        else:
            
            with st.expander("⚙️ Kies Analyse Periode (Dag, Maand of Jaar)", expanded=True):

                # Gebruik dezelfde logica als Tab 4 om de periode te selecteren
                clima_analysis_type = st.radio(
                    "Kies Analyse Niveau:",
                    ["Dag", "Maand", "Jaar"], 
                    key="clima_analysis_level"
                )
                
                # Bepaal het zoekbereik
                min_date_hist = df_daily_summary.index.min().date()
                max_date_hist = df_daily_summary.index.max().date()
                
                selected_period_str_clima = None
                df_clima_filter_base = df_daily_summary.copy()
                
                if clima_analysis_type == "Dag":
                    
                    default_date = max_date_hist
                         
                    selected_date = st.date_input(
                        "Kies de Analyse Datum:",
                        value=default_date,
                        min_value=min_date_hist,
                        max_value=max_date_hist,
                        key="clima_analysis_day_select"
                    )
                    
                    # Filter de dagelijkse samenvatting op de gekozen datum
                    df_clima_filter_base = df_clima_filter_base[
                        df_clima_filter_base.index.date == selected_date
                    ]

                    # De benchmark data wordt gefilterd op de maand-dag combinatie
                    selected_period_str_clima = selected_date.strftime('%d-%m-%Y')
                    clima_month_day = selected_date.strftime('%m-%d')


                elif clima_analysis_type == "Maand":
                    df_clima_filter_base['JaarMaand'] = df_clima_filter_base.index.to_period('M').astype(str)
                    available_periods = sorted(df_clima_filter_base['JaarMaand'].unique(), reverse=True)
                    titel_periode = "Jaar/Maand"

                    selected_period_str_clima = st.selectbox(
                        f"Kies de Analyse {titel_periode}:",
                        available_periods,
                        index=0,
                        key="clima_analysis_period_select_month"
                    )

                    df_clima_filter_base = df_clima_filter_base[df_clima_filter_base['JaarMaand'] == selected_period_str_clima]
                    df_clima_filter_base = df_clima_filter_base.drop(columns=['JaarMaand'])
                    
                else: # Jaar
                    df_clima_filter_base['Jaar'] = df_clima_filter_base.index.year
                    available_periods = sorted(df_clima_filter_base['Jaar'].unique(), reverse=True)
                    titel_periode = "Jaar"

                    selected_period_str_clima = st.selectbox(
                        f"Kies de Analyse {titel_periode}:",
                        available_periods,
                        index=0,
                        key="clima_analysis_period_select_year"
                    )

                    df_clima_filter_base = df_clima_filter_base[df_clima_filter_base['Jaar'] == selected_period_str_clima]
                    df_clima_filter_base = df_clima_filter_base.drop(columns=['Jaar'])
                

            st.subheader(f"Analyse: **{selected_period_str_clima}**")
            
            # --- De huidige periode data ---
            
            # Voor dag, halen we de werkelijke meetwaarden op van die dag
            if clima_analysis_type == "Dag":
                df_huidige_data = df_clima_filter_base.reset_index().set_index('Station Naam').copy()
                
                df_huidige_data = df_huidige_data.rename(columns={
                     'Temp_High_C': 'Abs. Max Temp (°C)',
                     'Temp_Low_C': 'Abs. Min Temp (°C)',
                     'Temp_Avg_C': 'Gem. Temp Periode (°C)',
                })
                
                df_huidige_data['Type'] = f"Huidige Dag: {selected_period_str_clima}"

            # Voor maand/jaar, berekenen we het gemiddelde over de dagen
            else:
                df_huidige_data = df_clima_filter_base.groupby('Station Naam').agg(
                    Max_Temp_Abs=('Temp_High_C', 'max'),
                    Min_Temp_Abs=('Temp_Low_C', 'min'),
                    Avg_Temp=('Temp_Avg_C', 'mean'),
                ).reset_index().set_index('Station Naam')
                
                df_huidige_data = df_huidige_data.rename(columns={
                    'Max_Temp_Abs': 'Abs. Max Temp (°C)',
                    'Min_Temp_Abs': 'Abs. Min Temp (°C)',
                    'Avg_Temp': 'Gem. Temp Periode (°C)',
                })
                
                df_huidige_data['Type'] = f"Huidige {clima_analysis_type}: {selected_period_str_clima}"
                
            
            # --- De Benchmark data van alle periodes ---
            
            all_benchmark_stats = []
            
            # all_hist_benchmarks is gesorteerd van oud naar nieuw (door de functie fetch_all_historical_benchmarks)
            for period_name, df_hist in all_hist_benchmarks.items():
                
                df_hist_copy = df_hist.copy()
                
                if clima_analysis_type == "Dag":
                    
                    df_period_data = df_hist_copy[df_hist_copy['Date'].dt.strftime('%m-%d') == clima_month_day]
                    
                    if not df_period_data.empty:
                         # Gemiddelde van die specifieke dag door alle jaren heen
                         # FIX: Gebruik alleen de numerieke kolommen om de TypeError te voorkomen
                         period_stats = df_period_data[['Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C']].mean().to_dict()
                         
                         df_stats = pd.DataFrame([{
                             'Abs. Max Temp (°C)': period_stats.get('Temp_High_C'),
                             'Abs. Min Temp (°C)': period_stats.get('Temp_Low_C'),
                             'Gem. Temp Periode (°C)': period_stats.get('Temp_Avg_C'),
                             'Type': f"Benchmark: {period_name}",
                             'Station Naam': 'Langjarig Gemiddelde'
                         }]).set_index('Station Naam')
                         all_benchmark_stats.append(df_stats)
                    
                
                elif clima_analysis_type == "Maand":
                    
                    # Filter op de gekozen maand
                    # OPMERKING: Bij Maand is 'selected_period_str_clima' YYYY-MM. We willen alleen MM
                    clima_month = selected_period_str_clima.split('-')[1]
                    df_period_data = df_hist_copy[df_hist_copy['Date'].dt.strftime('%m') == clima_month]
                    
                    if not df_period_data.empty:
                        # Gemiddelde van de dagelijkse Min, Max, Gem. over die maand, door alle 30 jaren heen
                        period_stats = df_period_data.agg({
                            'Temp_High_C': 'mean', # Gemiddelde van de Maxima op dagbasis over die maand
                            'Temp_Low_C': 'mean',  # Gemiddelde van de Minima op dagbasis over die maand
                            'Temp_Avg_C': 'mean',  # Gemiddelde van de Gemiddeldes op dagbasis over die maand
                        }).to_dict()
                        
                        df_stats = pd.DataFrame([{
                            'Abs. Max Temp (°C)': period_stats.get('Temp_High_C'),
                            'Abs. Min Temp (°C)': period_stats.get('Temp_Low_C'),
                            'Gem. Temp Periode (°C)': period_stats.get('Temp_Avg_C'),
                            'Type': f"Benchmark: {period_name}",
                            'Station Naam': 'Langjarig Gemiddelde'
                        }]).set_index('Station Naam')
                        all_benchmark_stats.append(df_stats)

                else: # Jaar
                    
                    if not df_hist_copy.empty:
                        # Gemiddelde van de dagelijkse Min, Max, Gem. over het hele jaar, door alle 30 jaren heen
                        period_stats = df_hist_copy.agg({
                            'Temp_High_C': 'mean',
                            'Temp_Low_C': 'mean',
                            'Temp_Avg_C': 'mean',
                        }).to_dict()
                        
                        df_stats = pd.DataFrame([{
                            'Abs. Max Temp (°C)': period_stats.get('Temp_High_C'),
                            'Abs. Min Temp (°C)': period_stats.get('Temp_Low_C'),
                            'Gem. Temp Periode (°C)': period_stats.get('Temp_Avg_C'),
                            'Type': f"Benchmark: {period_name}",
                            'Station Naam': 'Langjarig Gemiddelde'
                        }]).set_index('Station Naam')
                        all_benchmark_stats.append(df_stats)

            
            # --- Combineer en presenteer de resultaten ---
            
            df_clima_results_list = [df_huidige_data.reset_index()]
            df_clima_results_list.extend([df.reset_index() for df in all_benchmark_stats])
            
            if df_clima_results_list:
                df_clima_combined = pd.concat(df_clima_results_list, ignore_index=True)
                
                # Sorteervolgorde: Huidig > Nieuwste Benchmark > ... > Oudste Benchmark
                period_names_sorted_for_display = [
                    f"Huidige {clima_analysis_type}: {selected_period_str_clima}"
                ] + [f"Benchmark: {period}" for period in reversed(list(all_hist_benchmarks.keys()))]
                
                period_order_map = {name: i for i, name in enumerate(period_names_sorted_for_display)}
                df_clima_combined['Sort Order'] = df_clima_combined['Type'].map(period_order_map)
                
                df_clima_combined = df_clima_combined.sort_values(by=['Station Naam', 'Sort Order']).drop(columns=['Sort Order'])
                
                
                # Formatteer de kolommen
                for col in ['Abs. Max Temp (°C)', 'Abs. Min Temp (°C)', 'Gem. Temp Periode (°C)']:
                    df_clima_combined[col] = df_clima_combined[col].apply(lambda x: f"{x:.1f} °C" if pd.notna(x) else "N/A")
                
                df_clima_final_display = df_clima_combined.rename(columns={'Type': 'Analyse Type'}).set_index(['Station Naam', 'Analyse Type'])
                
                # Toon per Station Naam (Langjarig Gemiddelde is ook een 'station' nu)
                
                for station, df_group in df_clima_final_display.groupby('Station Naam'):
                    st.markdown(f"##### 📌 Station: **{station}**")
                    df_to_show = df_group.reset_index().drop(columns=['Station Naam']).set_index('Analyse Type')
                    
                    # Verwijder kolommen die niet met temperatuur te maken hebben
                    cols_to_keep = [col for col in df_to_show.columns if 'Temp' in col]
                    st.dataframe(df_to_show[cols_to_keep], use_container_width=True)
                    st.markdown("---")
            else:
                st.warning("Geen resultaten gevonden voor deze analyse.")
