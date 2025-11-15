import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime

# 1. Configuratie en constanten (Constants and Configuration)

# ===============================================================================
# Dit is de basis-URL voor de map /weatherdata/ op GitHub.
# ===============================================================================
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Pillmaster/ericmeteo/main/weatherdata/"

# Het jaar waar de data in de repository begint.
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


# 2. Functies voor Data (Loading & Processing)

# Zoekt naar beschikbare jaren op GitHub (Gecached voor 1 uur)
@st.cache_data(ttl=3600, show_spinner="Zoeken naar beschikbare jaren op GitHub...")
def discover_available_years(start_year, station_id, github_base_url):
    """
    Probeert iteratief om weather_YYYY.csv te laden van het startjaar tot het huidige jaar.
    """
    current_year = datetime.datetime.now().year
    available_years = []
    
    for year in range(start_year, current_year + 1):
        file_path = f"{station_id}/weather_{year}.csv"
        full_url = os.path.join(github_base_url, file_path).replace('\\', '/')
        
        try:
            # Laad alleen de header en de eerste rij (nrows=1) voor een snelle check
            df_test = pd.read_csv(full_url, sep=';', on_bad_lines='skip', nrows=1)
            
            # Controleer of het DataFrame niet leeg is en de verwachte kolom bevat
            if not df_test.empty and 'datum_waarneming_UTC' in df_test.columns:
                available_years.append(year)
            
        except Exception:
            pass
            
    return sorted(available_years)


# Bestaande Laadfunctie (past zich nu aan op de dynamische lijst van jaren)
@st.cache_data
def load_data(station_id, years, github_base_url, station_map, target_timezone):
    """
    Laadt, parseert en pre-verwerkt weerdata van GitHub voor meerdere jaren.
    """
    all_years_data = []
    station_name = station_map.get(station_id, station_id)

    for year in years:
        file_path = f"{station_id}/weather_{year}.csv"
        full_url = os.path.join(github_base_url, file_path).replace('\\', '/')

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
            # Log een fout als een specifiek jaar niet geladen kan worden
            st.warning(f"❌ Bestand niet gevonden of fout bij laden voor {station_name} in {year}. Reden: {e}")

    if all_years_data:
        df_station = pd.concat(all_years_data, ignore_index=True)
        return df_station.sort_values('Timestamp_Local').copy() 
    else:
        return pd.DataFrame() 


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


def find_consecutive_periods(df_filtered, min_days, temp_column):
    """Vindt en retourneert aaneengesloten periodes in een gefilterde DataFrame."""
    
    if df_filtered.empty:
        return pd.DataFrame(), 0

    df_groups = df_filtered.copy()
    
    # Zorg ervoor dat de index een DatetimeIndex is (Dagelijkse Samenvatting heeft dit al als index)
    if not isinstance(df_groups.index, pd.DatetimeIndex):
         return pd.DataFrame(), 0 
    
    # Reset index, sorteer en bepaal periodes per station
    df_groups = df_groups.reset_index().set_index('Date').sort_index()

    # Bepaal groepen per station
    grouped = df_groups.groupby('Station Naam')
    all_periods = []

    for name, group in grouped:
        group['new_period'] = (group.index.to_series().diff().dt.days.fillna(0) > 1).astype(int)
        group['group_id'] = group['new_period'].cumsum()
        
        # Aggregatie
        periods = group.groupby('group_id').agg(
            StartDatum=('Date', 'min'),
            EindDatum=('Date', 'max'),
            Duur=('Date', 'size'),
            Gemiddelde_Temp_Periode=(temp_column, 'mean')
        ).reset_index(drop=True)
        
        periods['Station Naam'] = name
        
        # Filter op de minimale duur
        periods = periods[periods['Duur'] >= min_days]
        all_periods.append(periods)

    if not all_periods:
        return pd.DataFrame(), 0

    periods_combined = pd.concat(all_periods, ignore_index=True)
    
    # Formatteer voor weergave
    periods_combined['StartDatum'] = periods_combined['StartDatum'].dt.strftime('%d-%m-%Y')
    periods_combined['EindDatum'] = periods_combined['EindDatum'].dt.strftime('%d-%m-%Y')
    
    # Gebruik safe_format_temp met .map() om TypeErrors te voorkomen
    periods_combined['Gemiddelde_Temp_Periode'] = periods_combined['Gemiddelde_Temp_Periode'].map(safe_format_temp)

    total_periods = len(periods_combined)
    
    return periods_combined.reset_index(drop=True), total_periods


# -------------------------------------------------------------------
# GECORRIGEERDE FUNCTIE: Extremen Analyse
# -------------------------------------------------------------------
def find_extreme_days(df_daily_summary, top_n=5):
    """
    Vindt de warmste (Max en Min Temp), koudste (Max en Min Temp), meest extreme 
    (grootste dagelijkse range) dagen en de hoogste/laagste GEMIDDELDE temperaturen 
    uit de dagelijkse samenvatting.
    """
    if df_daily_summary.empty:
        return {}

    # 1. Bereken de dagelijkse temperatuur range
    df_analysis = df_daily_summary.copy()
    df_analysis['Temp_Range_C'] = df_analysis['Temp_High_C'] - df_analysis['Temp_Low_C']

    results = {}

    # Definities van extremen (key: (kolom, ascending (True=oplopend/koud), display_col_focus))
    extremes_config = {
        # Warmte Extremen
        'hoogste_max_temp': ('Temp_High_C', False, 'Max Temp (°C)'),  # Warmste dag (Max Temp Top 5)
        'hoogste_min_temp': ('Temp_Low_C', False, 'Min Temp (°C)'),   # Warmste nacht (Min Temp Top 5)
        'hoogste_gem_temp': ('Temp_Avg_C', False, 'Gem Temp (°C)'),   # NIEUW: Warmste dag (Gem Temp Top 5)
        
        # Koude Extremen
        'laagste_min_temp': ('Temp_Low_C', True, 'Min Temp (°C)'),    # Koudste nacht (Min Temp Bottom 5)
        'laagste_max_temp': ('Temp_High_C', True, 'Max Temp (°C)'),   # Koudste dag (Max Temp Bottom 5)
        'laagste_gem_temp': ('Temp_Avg_C', True, 'Gem Temp (°C)'),    # NIEUW: Koudste dag (Gem Temp Bottom 5)
        
        # Range Extreem
        'grootste_range': ('Temp_Range_C', False, 'Range (°C)')
    }

    for key, (column, ascending, display_col) in extremes_config.items():
        
        # Groepeer per station en pak de top N (of bottom N via ascending)
        extreme_df_list = []
        for station, group in df_analysis.groupby('Station Naam'):
            
            top_days = group.sort_values(by=column, ascending=ascending).head(top_n)

            # 1. Maak de display DataFrame. 'Date' komt uit de index via reset_index()
            df_display = top_days.reset_index().copy()

            # 2. Formatteer de datum EERST, maak een NIEUWE kolom 'Datum' aan, en verwijder de oude 'Date' kolom
            if 'Date' in df_display.columns:
                 # Maak de geformatteerde kolom 'Datum' (deze is nu gegarandeerd aanwezig)
                 df_display['Datum'] = df_display['Date'].dt.strftime('%d-%m-%Y')
                 # Verwijder de oude kolom 'Date'
                 df_display = df_display.drop(columns=['Date']) 
            
            # 3. Definieer de renames (van interne, numerieke kolomnamen naar display namen)
            rename_dict = {
                'Temp_High_C': 'Max Temp (°C)', 
                'Temp_Low_C': 'Min Temp (°C)',
                'Temp_Avg_C': 'Gem Temp (°C)',
                'Temp_Range_C': 'Range (°C)' 
            }
            
            # Expliciet hernoemen van de numerieke kolommen
            df_display = df_display.rename(columns=rename_dict)
            
            # 4. Formatteer voor weergave (gebruik safe_format_temp)
            # Pas de lijst van te formatteren kolommen aan
            temp_display_cols = ['Max Temp (°C)', 'Min Temp (°C)', 'Range (°C)', 'Gem Temp (°C)']
            for col_name in temp_display_cols:
                if col_name in df_display.columns:
                     # Formatteer de numerieke waarden in de hernoemde kolom
                     df_display[col_name] = df_display[col_name].map(safe_format_temp)


            # 5. Herschikking van kolommen: Toon enkel de focus-kolom (tenzij het Range is)
            if key in ['hoogste_max_temp', 'laagste_max_temp']:
                 # Toon enkel Max Temp
                 final_cols_order = ['Datum', 'Station Naam', 'Max Temp (°C)']
            elif key in ['hoogste_min_temp', 'laagste_min_temp']:
                 # Toon enkel Min Temp
                 final_cols_order = ['Datum', 'Station Naam', 'Min Temp (°C)']
            elif key in ['hoogste_gem_temp', 'laagste_gem_temp']:
                 # Toon enkel Gem Temp
                 final_cols_order = ['Datum', 'Station Naam', 'Gem Temp (°C)']
            else: # grootste_range
                 # Toon alle kolommen (Range, Max en Min)
                 final_cols_order = ['Datum', 'Station Naam', 'Range (°C)', 'Max Temp (°C)', 'Min Temp (°C)']
            
            # Filter op de gewenste eindkolomvolgorde
            df_final_display = df_display[[c for c in final_cols_order if c in df_display.columns]].copy()
            
            extreme_df_list.append(df_final_display)

        if extreme_df_list:
             # Sorteer op Station Naam voor een consistente weergave
             results[key] = pd.concat(extreme_df_list, ignore_index=True).sort_values(by='Station Naam', ascending=True)

    return results

# -------------------------------------------------------------------
# HULPFUNCTIE: Voor een nette weergave in Tab 5
# -------------------------------------------------------------------
def display_extreme_results_by_station(df_results, title, info_text):
    """Toont extreme dagen, gegroepeerd per station met een duidelijke kop."""
    st.markdown(f"### {title}")
    st.info(info_text)
    
    if df_results is None or df_results.empty:
        st.warning("Geen data gevonden voor deze extreme categorie.")
        return

    # Groepeer op Station Naam om de tabellen te splitsen
    for station_name, df_station in df_results.groupby('Station Naam'):
        st.markdown(f"##### 📌 Station: **{station_name}**")
        # Zet 'Datum' als index voor een nette weergave, toon enkel de relevante kolommen
        display_cols = [c for c in df_station.columns if c not in ['Station Naam']]
        st.dataframe(df_station[display_cols].set_index('Datum'), use_container_width=True)
        st.markdown("---") # Visuele scheiding tussen de stations


# 3. Streamlit Applicatie Hoofdsectie (Streamlit Application Main)

st.set_page_config(
    page_title="Weergrafieken & Datazoeker",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(f"☀️ Weergrafieken ")
st.markdown("Analyseer en visualiseer weerdata van de Malmån stations.")

# --- Sidebar Controls (Definieer alle controls VOOR de data loading logica) ---

years_info_to_display = [] 
button_action = False 
status_placeholder = None 
info_placeholder_start_year = None
info_placeholder_years = None
info_placeholder_last_check = None

# 💥 Sectie 1: Data Basis (Stations & Jaren)
with st.sidebar.expander("🛠️ Data Basis (Stations & Jaren)", expanded=True):
    st.header("Stations & Jaren")

    station_options = list(STATION_MAP.keys())
    selected_station_ids = st.multiselect(
        "Kies één of meer weerstations:",
        options=station_options,
        default=[station_options[0]] if station_options else [], 
        format_func=lambda x: STATION_MAP[x]
    )

# 💥 Sectie 2: Grafiek Variabelen (Wordt later gedefinieerd)
# 💥 Sectie 3: Tijdsselectie (Wordt later gedefinieerd)
# 💥 Sectie 4: Historische Zoekfilters (Wordt later gedefinieerd)
# 💥 NIEUW: Sectie 5: Maand/Jaar Filters (Wordt later gedefinieerd)

# 💥 Sectie 6: Programma Checks (Status en Herlaadknop verplaatst HIER)
with st.sidebar.expander("⚙️ Programma Checks", expanded=False):
    st.header("Systeem Status & Controles")
    
    # Placeholder voor de dynamische statusmeldingen
    status_placeholder = st.empty() 

    # Herlaad Knop
    if st.button("Herlaad Data (Wis Cache)", key="reload_button_check"):
        st.cache_data.clear()
        button_action = True 
    
    st.markdown("---")
    
    # Info over gevonden jaren
    info_placeholder_start_year = st.empty()
    info_placeholder_years = st.empty()
    info_placeholder_last_check = st.empty()


# --- Data Loading Logic (Gebruikt de placeholders van hierboven) ---

if button_action:
    # Trigger een herstart na het wissen van de cache
    st.rerun() 
    
df_combined = pd.DataFrame()
failed_stations = []

if selected_station_ids:
    
    # 1. Jaarontdekking
    if status_placeholder:
        status_placeholder.info("Zoekt naar beschikbare datajaren op GitHub...", icon="⏳")
        
    # Gebruik de eerste geselecteerde ID om de jaren te ontdekken
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
    # Geen stations geselecteerd
    if status_placeholder:
        status_placeholder.info("Selecteer stations om data te laden.", icon="ℹ️")
    if info_placeholder_start_year:
        info_placeholder_start_year.markdown(f"**Start Jaar Zoektocht:** `{START_YEAR}`")
    if info_placeholder_years:
        info_placeholder_years.markdown(f"**Gevonden Jaarbestanden:** Geen. Selecteer stations.")
    if info_placeholder_last_check:
        info_placeholder_last_check.markdown(f"**Laatste Data Check:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Verwerk Data naar Dagelijkse Samenvatting (Voor Historische Analyse & Maand/Jaar Analyse) ---
df_daily_summary = pd.DataFrame()
if not df_combined.empty:
    
    df_combined_indexed = df_combined.set_index('Timestamp_Local')

    df_daily_summary = df_combined_indexed.groupby('Station Naam').resample('D').agg(
        Temp_High_C=('temp', 'max'),
        Temp_Low_C=('temp', 'min'),
        Temp_Avg_C=('temp', 'mean'),
        Pres_Avg_hPa=('druk', 'mean'), # Luchtdruk wordt gemiddeld
        Hum_Avg_P=('luchtvocht', 'mean'), # Vochtigheid wordt gemiddeld
    ).dropna(subset=['Temp_Avg_C']).reset_index()

    # Zorg ervoor dat de index de datum is voor de filtering en resampling
    df_daily_summary = df_daily_summary.rename(columns={'Timestamp_Local': 'Date'}).set_index('Date')
    df_daily_summary.index.name = 'Date' 

# Controleer of er data is geladen
if not df_combined.empty:
    
    # 💥 Sectie 2: Grafiek Variabelen
    with st.sidebar.expander("📈 Grafiek Variabelen", expanded=False):
        st.header("Grafiek Opties")

        plot_options = [COL_DISPLAY_MAP[col] for col in NUMERIC_COLS if col in df_combined.columns]
        
        default_selection = []
        temp_display_name = COL_DISPLAY_MAP.get('temp')
        
        if temp_display_name in plot_options:
            default_selection = [temp_display_name]
            
        # --- AANPASSING HIER: st.multiselect vervangen door st.selectbox ---
        if default_selection:
             default_index = plot_options.index(default_selection[0])
        else:
             default_index = 0
             
        selected_variable_display = st.selectbox(
            "Kies de variabele voor de grafiek:",
            options=plot_options,
            index=default_index, 
            key='variable_select_' + '_'.join(selected_station_ids) 
        )
        
        # selected_variables is nu een lijst met slechts één element
        selected_variables = [DISPLAY_TO_COL_MAP[selected_variable_display]]
        
        st.markdown(f"**Geselecteerde variabele:** `{selected_variable_display}`")


    # 💥 Sectie 3: Tijdsselectie (Live/Recent Data)
    with st.sidebar.expander("⏱️ Tijdsbereik (Grafiek/Ruwe Data)", expanded=False):
        st.header("Tijdsselectie")
        
        time_range_options = [
            "Huidige dag (sinds 00:00 uur)",
            "Laatste 24 uur",
            "Huidige maand",
            "Huidig jaar",
            "Vrije selectie op datum"
        ]

        selected_range_option = st.selectbox(
            "Kies een tijdsbereik voor Grafiek/Ruwe Data:",
            options=time_range_options,
            index=0 
        )

        now_local = pd.Timestamp.now(tz=TARGET_TIMEZONE)
        min_date_available = df_combined['Timestamp_Local'].min().date()
        max_date_available = df_combined['Timestamp_Local'].max().date()
        
        start_date_local = None
        end_date_local = now_local 

        if selected_range_option == "Huidige dag (sinds 00:00 uur)":
            start_date_local = now_local.normalize()
        
        elif selected_range_option == "Laatste 24 uur":
            start_date_local = now_local - pd.Timedelta(hours=24)
            
        elif selected_range_option == "Huidige maand":
            start_date_local = now_local.to_period('M').start_time.tz_localize(TARGET_TIMEZONE)
            
        elif selected_range_option == "Huidig jaar":
            start_date_local = now_local.to_period('Y').start_time.tz_localize(TARGET_TIMEZONE)
            
        elif selected_range_option == "Vrije selectie op datum":
            
            custom_date_range = st.date_input(
                "Kies start- en einddatum:",
                value=[max_date_available, max_date_available], 
                min_value=min_date_available,
                max_value=max_date_available,
                key='custom_date_selector_tab12'
            )
            
            if len(custom_date_range) == 2:
                start_date_local = pd.to_datetime(custom_date_range[0]).tz_localize(TARGET_TIMEZONE).normalize()
                end_date_local = pd.to_datetime(custom_date_range[1]).tz_localize(TARGET_TIMEZONE).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
            elif len(custom_date_range) == 1:
                st.warning("Selecteer een start- en einddatum.")
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
            
    
    # 💥 Sectie 4: Historische Zoekfilters
    with st.sidebar.expander("❄️ Historische Zoekfilters", expanded=False):
        
        st.markdown(f"**1. Zoekperiode**")
        period_options = ["Onbeperkt (Volledige Database)", "Selecteer Jaar", "Selecteer Maand", "Aangepaste Datums"]
        period_type = st.selectbox("1. Zoekperiode (Historisch)", period_options, key="period_select_hist")
        
        df_filtered_time = df_daily_summary.copy()
        
        min_date_hist = df_daily_summary.index.min().date()
        max_date_hist = df_daily_summary.index.max().date()
        
        if period_type == "Selecteer Jaar":
            available_years_hist = sorted(df_daily_summary.index.year.unique())
            st.selectbox("Kies Jaar:", available_years_hist, key="hist_year")
            df_filtered_time = df_daily_summary[df_daily_summary.index.year == st.session_state.hist_year]
        
        elif period_type == "Selecteer Maand":
            df_temp_filter = df_daily_summary.copy()
            df_temp_filter['JaarMaand'] = df_temp_filter.index.to_period('M').astype(str)
            available_periods = sorted(df_temp_filter['JaarMaand'].unique(), reverse=True)
            
            selected_period_str = st.selectbox("Kies Jaar en Maand (YYYY-MM):", available_periods, key="hist_year_month")
            
            df_filtered_time = df_temp_filter[df_temp_filter['JaarMaand'] == selected_period_str].drop(columns=['JaarMaand'])

        elif period_type == "Aangepaste Datums":
            date_range_hist = st.date_input(
                "Kies Start- en Einddatum:",
                value=(min_date_hist, max_date_hist),
                min_value=min_date_hist,
                max_value=max_date_hist,
                key="hist_dates"
            )
            
            if len(date_range_hist) == 2:
                df_filtered_time = df_daily_summary.loc[str(date_range_hist[0]):str(date_range_hist[1])]

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
                "Meetwaarde:", temp_type_options, index=2, key="temp_type_period" 
            )
            temp_column = temp_type.split(" (")[1][:-1] if " (" in temp_type else 'Temp_Avg_C'
            
            comparison = st.radio("Vergelijking:", ["Hoger dan (>=)", "Lager dan (<=)"], key="comparison_period") 
            
            temp_threshold = st.number_input(
                "Temperatuur (°C):", value=15.0, step=0.5, key="temp_threshold_period"
            )

        else: # Losse Dagen
            st.markdown("---")
            st.markdown(f"**4. Temperatuurfilter**")
            temp_type_options = ["Max Temp (Temp_High_C)", "Min Temp (Temp_Low_C)", "Gemiddelde Temp (Temp_Avg_C)"]
            temp_type = st.selectbox(
                "Meetwaarde:", temp_type_options, index=2, key="temp_type_days"
            )
            temp_column = temp_type.split(" (")[1][:-1] if " (" in temp_type else 'Temp_Avg_C'
            
            comparison = st.radio("Vergelijking:", ["Hoger dan (>=)", "Lager dan (<=)"], key="comparison_days")
            
            temp_threshold = st.number_input(
                "Temperatuur (°C):", value=15.0, step=0.5, key="temp_threshold_days"
            )
            
    # 💥 NIEUW: Sectie 5: Maand/Jaar Filters (Voor de nieuwe tab)
    with st.sidebar.expander("⚙️ Maand/Jaar Filters", expanded=True):
        st.header("Analyse Periode")

        analysis_type = st.radio(
            "Kies Analyse Niveau:",
            ["Maand", "Jaar"],
            key="analysis_level_new"
        )
        
        # We moeten de data per station groeperen en dan de unieke periodes vinden
        if df_daily_summary.empty:
            st.info("Laad eerst data om periodes te selecteren.")
            selected_period_str = None
            df_analysis_selector = pd.DataFrame()
            titel_periode = "Geen Data"
        else:
            # Gebruik de data van alle geselecteerde stations (hoewel de analyse hieronder station-afhankelijk wordt)
            df_analysis_selector = df_daily_summary.copy()
            
            if analysis_type == "Maand":
                df_analysis_selector['JaarMaand'] = df_analysis_selector.index.to_period('M').astype(str)
                # Unieke periodes over alle geselecteerde stations
                available_periods = sorted(df_analysis_selector['JaarMaand'].unique(), reverse=True)

                selected_period_str = st.selectbox(
                    "Selecteer Maand:",
                    available_periods,
                    key="monthly_selector_new"
                )
                if selected_period_str:
                    selected_period = pd.to_datetime(selected_period_str)
                    titel_periode = selected_period.strftime('%B %Y')
                else:
                    titel_periode = "Selecteer Maand"
                
            else: # Jaar
                available_years = sorted(df_analysis_selector.index.year.unique(), reverse=True)

                selected_year = st.selectbox(
                    "Selecteer Jaar:",
                    available_years,
                    key="yearly_selector_new"
                )
                selected_period_str = str(selected_year) if selected_year else None
                titel_periode = str(selected_year) if selected_year else "Selecteer Jaar"
        
        st.markdown(f"**Geselecteerde periode:** {titel_periode}")

    
    # --- Main Content: Tabs ---
    
    # FIX: 5 tabs gedefinieerd
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Grafiek", "🔍 Data Zoeken", "❄️ Historische Analyse", "🗓️ Maand/Jaar Analyse", "🔥 Extreme Dagen"])

    # -------------------------------------------------------------------
    # TAB 1: GRAFIEK (ALTAIR)
    # -------------------------------------------------------------------
    with tab1:
        st.header("Vergelijking Weergrafieken")
        
        if selected_variables and not filtered_df.empty:
            
            # --- NIEUWE LOGICA: Overzicht en Kerncijfers ---
            
            # Stap 1: Bepaal de numerieke kolommen voor aggregatie (temperatuur en druk)
            temp_cols = [col for col in ['temp', 'druk'] if col in filtered_df.columns]
            
            # Stap 2: Aggregeer de gefilterde ruwe data per station
            if temp_cols:
                 aggregation_dict = {
                     'temp': ['mean', 'max', 'min'], # Bereken Gem, Max en Min voor Temp
                     'druk': ['mean'] # Bereken Gemiddelde voor Druk
                 }
            
                 # Bepaal de indices voor max/min temperatuur (om de datum te vinden)
                 idx_max = filtered_df.groupby('Station Naam')['temp'].idxmax()
                 idx_min = filtered_df.groupby('Station Naam')['temp'].idxmin()

                 # Voer de aggregatie uit
                 analysis_results = filtered_df.groupby('Station Naam').agg(
                     avg_temp=('temp', 'mean'),
                     max_temp=('temp', 'max'),
                     min_temp=('temp', 'min'),
                     avg_pres=('druk', 'mean'),
                 ).reset_index()
                 
                 # Voeg de datums toe aan de resultaten (via merge op Station Naam)
                 if not idx_max.empty and not idx_min.empty:
                      df_max_date = filtered_df.loc[idx_max, ['Station Naam', 'Timestamp_Local']].rename(columns={'Timestamp_Local': 'max_temp_date'})
                      df_min_date = filtered_df.loc[idx_min, ['Station Naam', 'Timestamp_Local']].rename(columns={'Timestamp_Local': 'min_temp_date'})
                      
                      analysis_results = pd.merge(analysis_results, df_max_date, on='Station Naam', how='left')
                      analysis_results = pd.merge(analysis_results, df_min_date, on='Station Naam', how='left')
                      
                 # Weergave van de Kerncijfers
                 st.subheader("Overzicht en Kerncijfers (Geselecteerde Periode)")

                 for index, row in analysis_results.iterrows():
                    station_name = row['Station Naam']
                    
                    # 💥 NIEUWE LOGICA: Laatste Waarneming voor dit station en deze periode
                    df_station_filtered = filtered_df[filtered_df['Station Naam'] == station_name].sort_values('Timestamp_Local', ascending=False)
                    
                    last_temp = float('nan')
                    last_time_str = "N/A"

                    if not df_station_filtered.empty:
                        last_record = df_station_filtered.iloc[0]
                        last_temp = last_record['temp']
                        last_time_str = last_record['Timestamp_Local'].strftime('%d-%m-%Y %H:%M')
                    
                    st.markdown(f"#### 🏷️ Station: **{station_name}**")
                    # Gecorrigeerd naar 5 kolommen (col0 t/m col4)
                    col0, col1, col2, col3, col4 = st.columns(5) 
                    
                    # Datum formatteer HULP:
                    max_date_str = "N/A"
                    if 'max_temp_date' in row and pd.notna(row['max_temp_date']):
                         # FIX: Gebruik strftime direct op de pd.Timestamp-objecten
                         max_date_str = row['max_temp_date'].strftime('%d-%m-%Y %H:%M')
                    
                    min_date_str = "N/A"
                    if 'min_temp_date' in row and pd.notna(row['min_temp_date']):
                         # FIX: Gebruik strftime direct op de pd.Timestamp-objecten
                         min_date_str = row['min_temp_date'].strftime('%d-%m-%Y %H:%M')


                    with col0:
                        if pd.notna(last_temp):
                            st.metric(
                                label="Laatste Temp", 
                                value=f"{last_temp:.1f} °C", 
                                delta=f"Op: {last_time_str}"
                            )
                        else:
                            st.metric(label="Laatste Temp", value="N/A")

                    with col1:
                        st.metric(label="Gemiddelde Temp", value=f"{row['avg_temp']:.1f} °C")

                    with col2:
                        st.metric(
                            label="Hoogste Max Temp",
                            value=f"{row['max_temp']:.1f} °C",
                            delta=f"Op: {max_date_str}"
                        )

                    with col3:
                        st.metric(
                            label="Laagste Min Temp",
                            value=f"{row['min_temp']:.1f} °C",
                            delta=f"Op: {min_date_str}"
                        )

                    with col4:
                        st.metric(label="Gemiddelde Luchtdruk", value=f"{row['avg_pres']:.2f} hPa")

                    st.markdown("---")
            else:
                 st.info("Kerncijfers (temp/druk) zijn niet beschikbaar voor de geselecteerde variabele(n).")


            # --- Grafiek Logica (onveranderd) ---
            
            # selected_variables bevat nu maar één element [col_name]
            plot_df = filtered_df[['Timestamp_Local', 'Station Naam'] + selected_variables].rename(columns=COL_DISPLAY_MAP).melt(
                ['Timestamp_Local', 'Station Naam'], 
                var_name='Variabele Naam', 
                value_name='Waarde'
            )
            
            # De display naam van de geselecteerde variabele
            variable_name_display = plot_df['Variabele Naam'].iloc[0]
            
            base = alt.Chart(plot_df).encode(
                x=alt.X('Timestamp_Local', title=f"Tijd ({TARGET_TIMEZONE.split('/')[1]} - Zomer/Wintertijd)", axis=None), 
                y=alt.Y('Waarde', title=variable_name_display, scale=alt.Scale(zero=False)), # Gebruik de variabele naam als Y-as titel
                
                # FIX: Legenda terug naar rechts (werkt nu goed bij één plot)
                color=alt.Color('Station Naam', title="Station", legend=alt.Legend(orient="right")), 
                
                tooltip=[
                    alt.Tooltip('Timestamp_Local', title='Tijd (Lokaal)', format='%Y-%m-%d %H:%M:%S'), 
                    'Station Naam',
                    'Variabele Naam', 
                    alt.Tooltip('Waarde', format='.2f')
                ]
            ).properties(
                title=f"Weerdata Tijdreeks Vergelijking: {variable_name_display}", # Titel is nu duidelijker
            ).interactive() 
            
            # --- GEEN FACETING MEER (GEEN ROW=ALT.ROW) ---
            line_chart = base.mark_line().encode(
                x=alt.X(
                    'Timestamp_Local', 
                    title=f"Tijd ({TARGET_TIMEZONE.split('/')[1]} - Zomer/Wintertijd)",
                    axis=alt.Axis(format='%H:%M')
                ),
                opacity=alt.value(0.8)
            ) # .resolve_scale(x='independent', y='independent') is nu ook niet meer nodig
            
            # GEBRUIK use_container_width=True voor dynamische breedte
            st.altair_chart(line_chart, use_container_width=True)

        elif filtered_df.empty:
             st.warning("Er is geen data beschikbaar in het geselecteerde datumbereik.")
        else:
            st.warning("Selecteer één variabele in de zijbalk om een grafiek te maken.")


    # -------------------------------------------------------------------
    # TAB 2: RUWE DATA
    # -------------------------------------------------------------------
    with tab2:
        st.header("Ruwe Data Zoeken (Gefilterd)")
        st.markdown(f"Gefilterde gecombineerde data van **{date_range_display_start}** tot **{date_range_display_end}** (Lokale tijd).")
        
        if not filtered_df.empty:
            display_df = filtered_df.rename(columns={
                **COL_DISPLAY_MAP, 
                'Timestamp_Local': 'Tijd (Lokaal)',
                'Timestamp_UTC': 'Tijd (UTC)' 
            })
            
            relevant_cols = ['Station Naam', 'Tijd (Lokaal)'] + [COL_DISPLAY_MAP[col] for col in NUMERIC_COLS if COL_DISPLAY_MAP[col] in display_df.columns]
            
            st.dataframe(
                display_df[relevant_cols].set_index('Tijd (Lokaal)'), 
                use_container_width=True
            )
            
            st.download_button(
                label="Download Gefilterde Data als CSV",
                data=filtered_df.to_csv(index=False, sep=';').encode('utf-8'),
                file_name=f"weerdata_vergelijking_lokaal_{start_display.date()}_tot_{end_display.date()}.csv",
                mime="text/csv"
            )
        else:
            st.info("Geen data om weer te geven. Pas het tijdsbereik aan.")

    # -------------------------------------------------------------------
    # TAB 3: HISTORISCHE ANALYSE
    # -------------------------------------------------------------------
    with tab3:
        st.header("Historische Analyse (Dagelijkse Samenvatting)")
        
        if df_daily_summary.empty:
            st.warning("Geen dagelijkse samenvattingsdata beschikbaar voor analyse.")
            st.stop()
            
        # C. FILTEREN (Uitvoering)
        df_filtered_condition = df_filtered_time.copy()

        if filter_mode == "Hellmann Getal Berekenen":
            df_filtered_condition = df_filtered_condition[df_filtered_condition['Temp_Avg_C'] <= 0.0].copy()
        elif comparison == "Hoger dan (>=)":
            df_filtered_condition = df_filtered_condition[df_filtered_condition[temp_column] >= temp_threshold].copy()
        elif comparison == "Lager dan (<=)":
            df_filtered_condition = df_filtered_condition[df_filtered_condition[temp_column] <= temp_threshold].copy()
        
        
        # --- Resultaten Weergave ---
        
        st.subheader(f"Zoekopdracht Samenvatting (Station: {', '.join(selected_station_ids)})")
        
        search_summary_parts = []
        
        if not df_filtered_time.empty:
            min_res_date = df_filtered_time.index.min().strftime('%d-%m-%Y')
            max_res_date = df_filtered_time.index.max().strftime('%d-%m-%Y')
            search_summary_parts.append(f"**Periode:** {min_res_date} t/m {max_res_date}")
        
        if filter_mode == "Hellmann Getal Berekenen":
            search_summary_parts.append("**Modus:** Hellmann Getal Berekenen (Gem. Temp $\\le$ 0.0 °C)")
        
        elif filter_mode == "Aaneengesloten Periode":
            temp_label = st.session_state.temp_type_period.split(" (")[0]
            comp_symbol = ">=" if comparison == "Hoger dan (>=)" else "<="
            search_summary_parts.append(f"**Modus:** Aaneengesloten Periode ({min_consecutive_days}+ dagen)")
            search_summary_parts.append(f"**Drempel:** {temp_label} {comp_symbol} **{temp_threshold:.1f} °C**")
        
        elif filter_mode == "Losse Dagen":
            temp_label = st.session_state.temp_type_days.split(" (")[0]
            comp_symbol = ">=" if comparison == "Hoger dan (>=)" else "<="
            search_summary_parts.append("**Modus:** Losse Dagen")
            search_summary_parts.append(f"**Drempel:** {temp_label} {comp_symbol} **{temp_threshold:.1f} °C**")

        if search_summary_parts:
            st.info(" | ".join(search_summary_parts))
        st.markdown("---")


        if df_filtered_condition.empty and filter_mode != "Hellmann Getal Berekenen":
            st.info("Geen dagen gevonden die voldoen aan de ingestelde filters.")
            
        elif filter_mode == "Losse Dagen":
            df_final = df_filtered_condition
            
            st.subheader(f"Resultaten ({len(df_final)} losse dagen gevonden)")

            start_date_str = df_final.index.min().strftime('%d-%m-%Y')
            end_date_str = df_final.index.max().strftime('%d-%m-%Y')
            st.metric(
                label="Totaal aantal dagen gevonden:",
                value=f"{len(df_final)} dagen",
                delta=f"Periode: {start_date_str} t/m {end_date_str}"
            )
            st.subheader("Gevonden Dagen")
            df_display = df_final.copy().reset_index().rename(columns={'Date': 'Datum', 'Temp_High_C': 'Max Temp (°C)', 'Temp_Low_C': 'Min Temp (°C)', 'Temp_Avg_C': 'Gem Temp (°C)', 'Hum_Avg_P': 'Gem Vochtigheid (%)', 'Pres_Avg_hPa': 'Gem Druk (hPa)'})
            df_display['Datum'] = df_display['Datum'].dt.strftime('%d-%m-%Y')
            
            # Gebruik safe_format_temp voor weergave
            for col in ['Max Temp (°C)', 'Min Temp (°C)', 'Gem Temp (°C)']:
                 df_display[col] = df_display[col].map(safe_format_temp)


            st.dataframe(df_display.set_index('Datum')[['Station Naam', 'Max Temp (°C)', 'Min Temp (°C)', 'Gem Temp (°C)', 'Gem Vochtigheid (%)', 'Gem Druk (hPa)']], use_container_width=True)
            
            st.subheader("Temperatuurdistributie van de Gevonden Dagen")
            
            temp_label = st.session_state.temp_type_days.split(" (")[0]
            
            fig_hist = px.histogram(
                df_final, x=temp_column, nbins=30, 
                title=f"Temperatuurdistributie ({temp_label})",
                labels={temp_column: f"{temp_label} (°C)"}, 
                template="plotly_white",
                color='Station Naam' 
            )
            fig_hist.add_vline(x=temp_threshold, line_width=2, line_dash="dash", line_color="red", annotation_text="Drempel")
            st.plotly_chart(fig_hist, use_container_width=True)


        elif filter_mode == "Aaneengesloten Periode":
            
            df_periods, total_periods = find_consecutive_periods(df_filtered_condition, min_consecutive_days, temp_column)

            st.subheader(f"Resultaten ({total_periods} periodes gevonden)")

            if df_periods.empty:
                st.info(f"Geen aaneengesloten periodes van {min_consecutive_days} of meer dagen gevonden die aan de filters voldoen.")
            else:
                totaal_dagen = df_periods['Duur'].sum()
                st.metric(
                    label=f"Totaal {min_consecutive_days}+ dagen periodes gevonden:",
                    value=f"{total_periods} periodes",
                    delta=f"Totaal: {totaal_dagen} dagen"
                )
                df_periods.rename(columns={'StartDatum': 'Startdatum', 'EindDatum': 'Einddatum', 'Duur': 'Aantal Dagen'}, inplace=True)
                st.subheader("Gevonden Periodes")
                st.dataframe(df_periods, use_container_width=True)
                fig_bar = px.bar(
                    df_periods, x=df_periods.index, y='Aantal Dagen', color='Station Naam',
                    title="Gevonden Periodes per Station en Duur",
                    labels={'x': 'Periode Index', 'Aantal Dagen': 'Aantal Dagen'},
                    hover_data=['Startdatum', 'Einddatum', 'Gemiddelde_Temp_Periode'], template="plotly_white"
                )
                st.plotly_chart(fig_bar, use_container_width=True)


        elif filter_mode == "Hellmann Getal Berekenen":
            
            # 1. Hellmann-waarden per station berekenen
            df_hellmann_calc = df_filtered_condition.groupby('Station Naam').agg(
                Hellmann_Value=('Temp_Avg_C', 'sum'),
                Aantal_Vorstdagen=('Temp_Avg_C', 'size')
            ).reset_index()

            # De Hellmann-waarde is de absolute som van de negatieve gemiddelde dagtemperaturen
            df_hellmann_calc['Hellmann_Value'] = abs(df_hellmann_calc['Hellmann_Value'])
            
            # --- Resultaat: Tabel per station ---
            
            # We tonen de totale waarde NIET meer bovenaan.
            st.subheader("Hellmann Waarden per Station")
            
            if df_hellmann_calc.empty or df_hellmann_calc['Aantal_Vorstdagen'].sum() == 0:
                st.info("Geen dagen met een gemiddelde temperatuur $\\le$ 0.0 °C gevonden in de geselecteerde periode.")
            else:
                # Toon de samenvattingstabel per station
                st.dataframe(df_hellmann_calc, use_container_width=True)
                
                
                st.markdown("---")
                
                # --- Resultaat: Gedetailleerde Vorstdagen per station ---
                
                st.subheader("Gedetailleerde Vorstdagen per Station (Tgem $\\le$ 0.0 °C)")

                df_detail_display = df_filtered_condition.reset_index().rename(columns={
                    'Date': 'Datum', 
                    'Temp_Avg_C': 'Gem Temp (°C)', 
                    'Temp_Low_C': 'Min Temp (°C)'
                })
                
                # Bereid de kolommen voor op weergave
                # De Vorstbijdrage is de absolute waarde van de negatieve Gemiddelde Temp
                df_detail_display['Vorstbijdrage'] = df_detail_display['Gem Temp (°C)'] * (-1)
                
                # Formatteer de numerieke waarden
                for col in ['Min Temp (°C)', 'Gem Temp (°C)']:
                     df_detail_display[col] = df_detail_display[col].map(safe_format_temp)
                
                # Formatteer de Vorstbijdrage (met 3 decimalen voor precisie van de Hellmann-som)
                df_detail_display['Vorstbijdrage'] = df_detail_display['Vorstbijdrage'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")


                # Lus over de stations en toon de detailtabel gescheiden
                for station_name in df_detail_display['Station Naam'].unique():
                    
                    df_station_days = df_detail_display[df_detail_display['Station Naam'] == station_name].copy()
                    
                    # Haal de berekende Hellmann-waarde voor dit station op
                    hellmann_data = df_hellmann_calc[df_hellmann_calc['Station Naam'] == station_name]
                    if not hellmann_data.empty:
                         hellmann_value = hellmann_data['Hellmann_Value'].iloc[0]
                         vorst_dagen = hellmann_data['Aantal_Vorstdagen'].iloc[0]
                    else:
                         hellmann_value = 0.0
                         vorst_dagen = 0
                    
                    st.markdown(f"#### ❄️ Station: **{station_name}**")
                    st.metric(
                         label=f"Totale Hellmann-waarde voor {station_name}:",
                         value=f"{hellmann_value:.3f}",
                         delta=f"Aantal Vorstdagen: {vorst_dagen}"
                    )
                    
                    st.dataframe(
                        df_station_days.set_index('Datum')[['Min Temp (°C)', 'Gem Temp (°C)', 'Vorstbijdrage']], 
                        use_container_width=True
                    )
                    st.markdown("---")


    # -------------------------------------------------------------------
    # TAB 4: MAAND/JAAR ANALYSE
    # -------------------------------------------------------------------
    with tab4:
        st.header(f"🗓️ Gedetailleerde Analyse voor {titel_periode}")
        st.markdown("---")
        
        if df_daily_summary.empty:
            st.warning("Geen dagelijkse samenvattingsdata beschikbaar voor deze analyse. Zorg ervoor dat data is geladen.")
        elif selected_period_str is None or titel_periode == "Selecteer Jaar" or titel_periode == "Selecteer Maand":
            st.info("Selecteer eerst een **Maand** of **Jaar** in de zijbalk onder '⚙️ Maand/Jaar Filters'.")
        else:
            
            # Filter de dagelijkse samenvatting op de geselecteerde periode
            if analysis_type == "Maand":
                # Filter op de periode (YYYY-MM string)
                df_selected = df_daily_summary[df_daily_summary.index.to_period('M').astype(str) == selected_period_str]
            else: # Jaar
                # Filter op het jaar (integer)
                df_selected = df_daily_summary[df_daily_summary.index.year == int(selected_period_str)]
                
            if df_selected.empty:
                 st.warning(f"Geen data gevonden voor de geselecteerde periode: {titel_periode}.")
            else:
                 
                # Groepeer per station om de kerncijfers te berekenen
                analysis_results = df_selected.groupby('Station Naam').agg(
                    avg_temp=('Temp_Avg_C', 'mean'),
                    max_temp=('Temp_High_C', 'max'),
                    min_temp=('Temp_Low_C', 'min'),
                    avg_pres_month=('Pres_Avg_hPa', 'mean'),
                    max_temp_date=('Temp_High_C', 'idxmax'),
                    min_temp_date=('Temp_Low_C', 'idxmin')
                ).reset_index()

                st.subheader(f"Overzicht en Kerncijfers ({titel_periode})")
                
                for index, row in analysis_results.iterrows():
                    station_name = row['Station Naam']
                    
                    st.markdown(f"#### 🏷️ Station: **{station_name}**")
                    col1, col2, col3, col4 = st.columns(4) 
                    
                    # De datums zijn nu pd.Timestamp objecten door idxmax/idxmin
                    max_date_str = "N/A"
                    if pd.notna(row['max_temp_date']):
                         max_date_str = row['max_temp_date'].strftime('%d-%m-%Y')
                    
                    min_date_str = "N/A"
                    if pd.notna(row['min_temp_date']):
                         min_date_str = row['min_temp_date'].strftime('%d-%m-%Y')


                    with col1:
                        st.metric(label="Gemiddelde Temp", value=f"{row['avg_temp']:.1f} °C")

                    with col2:
                        st.metric(
                            label="Hoogste Max Temp",
                            value=f"{row['max_temp']:.1f} °C",
                            delta=f"Op: {max_date_str}"
                        )

                    with col3:
                        st.metric(
                            label="Laagste Min Temp",
                            value=f"{row['min_temp']:.1f} °C",
                            delta=f"Op: {min_date_str}"
                        )

                    with col4:
                        st.metric(label="Gemiddelde Luchtdruk", value=f"{row['avg_pres_month']:.2f} hPa")

                    st.markdown("---")
                
                
                st.subheader(f"Temperatuur Trend ({titel_periode})")
                
                # Plot de geselecteerde data (meerdere stations in één grafiek)
                fig_temp = go.Figure()

                for station in df_selected['Station Naam'].unique():
                    # FIX: Gebruik reset_index() om 'Date' als kolom beschikbaar te maken voor Plotly
                    df_plot = df_selected[df_selected['Station Naam'] == station].reset_index()
                    
                    fig_temp.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Temp_High_C'], mode='lines', name=f"Max ({station})", line=dict(dash='solid')))
                    fig_temp.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Temp_Avg_C'], mode='lines', name=f"Gem ({station})", line=dict(dash='dot')))
                    fig_temp.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Temp_Low_C'], mode='lines', name=f"Min ({station})", line=dict(dash='dash')))
                
                fig_temp.update_layout(
                    title=f"Temperatuurverloop in {titel_periode}",
                    xaxis_title="Datum", yaxis_title="Temperatuur (°C)", hovermode="x unified", template="plotly_white"
                )
                st.plotly_chart(fig_temp, use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 5: EXTREME DAGEN (Volledige Dataset)
    # -------------------------------------------------------------------
    with tab5:
        st.header(f"🔥 Extreme Dagen (Analyse over de Volledige Dataset)")
        st.markdown(f"Dit overzicht toont de Top 5 extremen van de gehele beschikbare database ({df_daily_summary.index.min().strftime('%d-%m-%Y')} t/m {df_daily_summary.index.max().strftime('%d-%m-%Y')}).")
        st.markdown("---")
        
        if df_daily_summary.empty:
            st.warning("Geen dagelijkse samenvattingsdata beschikbaar voor deze analyse.")
        else:
            
            # Bereken de extreme dagen met de gecorrigeerde functie over de volledige data
            extreme_results_full = find_extreme_days(df_daily_summary, top_n=5)
            
            # Nieuwe Sub-tabs voor de 7 categorieën
            tab_high_max, tab_high_min, tab_low_min, tab_low_max, tab_range, tab_high_avg, tab_low_avg = st.tabs([
                "🔥 Hoogste Max Temp", 
                "☀️ Hoogste Min Temp", 
                "🥶 Laagste Min Temp", 
                "🌬️ Laagste Max Temp", 
                "🌡️ Grootste Dagelijkse Range",
                "📈 Hoogste Gem Temp", 
                "📉 Laagste Gem Temp"
            ])
            
            # --- Hoogste Max Temp ---
            with tab_high_max:
                display_extreme_results_by_station(
                    extreme_results_full.get('hoogste_max_temp'), 
                    "Hoogste Maximum Temperatuur (Top 5 per station) - De Warmste Dagen", 
                    "De dagen met de absoluut hoogste gemeten temperatuur."
                )
            
            # --- Hoogste Min Temp (Warmste Nacht) ---
            with tab_high_min:
                display_extreme_results_by_station(
                    extreme_results_full.get('hoogste_min_temp'), 
                    "Hoogste Minimum Temperatuur (Top 5 per station) - De Warmste Nachten", 
                    "De dagen waarop het 's nachts/ochtends het warmst bleef (koelde het minst af)."
                )
            
            # --- Laagste Min Temp (Koudste Nacht) ---
            with tab_low_min:
                display_extreme_results_by_station(
                    extreme_results_full.get('laagste_min_temp'), 
                    "Laagste Minimum Temperatuur (Top 5 per station) - De Koudste Nachten", 
                    "De dagen waarop de temperatuur het diepst zakte."
                )
                    
            # --- Laagste Max Temp (Koudste Dag) ---
            with tab_low_max:
                display_extreme_results_by_station(
                    extreme_results_full.get('laagste_max_temp'), 
                    "Laagste Maximum Temperatuur (Top 5 per station) - De Koudste Dagen", 
                    "De dagen waarop de temperatuur overdag het laagst bleef (warmde het minst op)."
                )
            
            # --- Hoogste Gem Temp (NIEUW) ---
            with tab_high_avg:
                display_extreme_results_by_station(
                    extreme_results_full.get('hoogste_gem_temp'), 
                    "Hoogste Gemiddelde Temperatuur (Top 5 per station) - De Warmste Gemiddelde Dagen", 
                    "De dagen met het hoogste gemiddelde over 24 uur."
                )

            # --- Laagste Gem Temp (NIEUW) ---
            with tab_low_avg:
                display_extreme_results_by_station(
                    extreme_results_full.get('laagste_gem_temp'), 
                    "Laagste Gemiddelde Temperatuur (Top 5 per station) - De Koudste Gemiddelde Dagen", 
                    "De dagen met het laagste gemiddelde over 24 uur."
                )
                    
            # --- Grootste Range ---
            with tab_range:
                display_extreme_results_by_station(
                    extreme_results_full.get('grootste_range'), 
                    "Grootste Dagelijkse Temperatuurbereik (Top 5 Range per station)", 
                    "De Dagelijkse Range is het verschil tussen de Max Temp en de Min Temp op die dag (Grote schommelingen)."
                )
