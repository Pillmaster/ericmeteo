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

# Zoekt naar beschikbare jaren op GitHub (Gecached, NU ZONDER TTL VOOR ACTUEELHEID)
@st.cache_data(show_spinner="Zoeken naar beschikbare jaren op GitHub...")
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


# Bestaande Laadfunctie (caching behouden voor performantie)
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
        
# NIEUWE FUNCTIE: Haalt de eenheid uit de display-naam
def get_unit_from_display_name(display_name, plot_col):
    """
    Haalt de eenheid uit de display naam voor formatting.
    """
    if plot_col in ['temp', 'dauwpunt', 'natbol']:
        # Voor temperaturen voegen we zelf '°C' toe, dus return lege string
        return "°C" # Return °C zodat we het overal kunnen gebruiken voor metrics
    # Zoek naar tekst tussen haken (bijv. 'hPa' in 'Luchtdruk (hPa)')
    unit_match = re.search(r'\((.*?)\)', display_name)
    if unit_match:
        return unit_match.group(1).strip()
    # Als er geen haken zijn, probeer het laatste woord
    return display_name.split(' ')[-1].strip()


def find_consecutive_periods(df_filtered, min_days, temp_column):
    """Vindt en retourneert aaneengesloten periodes in een gefilterde DataFrame."""
    
    if df_filtered.empty:
        return pd.DataFrame(), 0

    df_groups = df_filtered.copy()
    
    # Zorg ervoor dat de index een DatetimeIndex is (Dagelijkse Samenvatting heeft dit al als index)
    if not isinstance(df_groups.index, pd.DatetimeIndex):
        return pd.DataFrame(), 0 
    
    # Correctie: Reset de index ZODAT 'Date' een reguliere kolom is, en sorteer
    # Dit lost de 'KeyError: Column(s) ['Date'] do not exist' op tijdens de aggregatie.
    df_groups = df_groups.reset_index().sort_values('Date') # <-- CORRECTIE 1: Zorg dat 'Date' een kolom is.

    # Bepaal groepen per station
    grouped = df_groups.groupby('Station Naam')
    all_periods = []

    for name, group in grouped:
        # Nu 'Date' een kolom is, gebruiken we deze voor de diff berekening
        group['new_period'] = (group['Date'].diff().dt.days.fillna(0) > 1).astype(int) # <-- CORRECTIE 2: Gebruik group['Date']
        group['group_id'] = group['new_period'].cumsum()
        
        # Aggregatie: 'Date' wordt nu correct gevonden als kolom.
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


# --- Session State Initialisatie & Callback Functie ---
# De index van de tab die actief moet zijn. 0 = Grafiek, 2 = Historische Analyse, 3 = Maand/Jaar Analyse
if 'active_tab_index' not in st.session_state:
    st.session_state.active_tab_index = 0

def set_active_tab(index):
    """Callback functie om de actieve tab in de hoofdsectie in te stellen."""
    st.session_state.active_tab_index = index
# -----------------------------------------------------

# --- Sidebar Controls (Definieer alle controls VOOR de data loading logica) ---

years_info_to_display = [] 
button_action = False 
status_placeholder = None 
info_placeholder_start_year = None
info_placeholder_years = None
info_placeholder_last_check = None

# 💥 Sectie 1: Data Basis (Stations & Jaren)
# STANDAARD OPEN: Meest essentiële keuze
with st.sidebar.expander("🛠️ Data Basis (Stations & Jaren)", expanded=True):
    # st.header("Stations & Jaren") # VERWIJDERD VOOR COMPACTHEID

    station_options = list(STATION_MAP.keys())
    selected_station_ids = st.multiselect(
        "Kies één of meer weerstations:",
        options=station_options,
        default=[station_options[0]] if station_options else [], 
        format_func=lambda x: STATION_MAP[x]
    )

# 💥 Sectie 2 & 3: Grafiek Opties (Variabelen & Tijd) - GECONSOLIDEERD
# 💥 Sectie 4: Historische Zoekfilters (Wordt later gedefinieerd)
# 💥 Sectie 5: Maand/Jaar Filters (Wordt later gedefinieerd)

# 💥 Sectie 6: Programma Checks (Status en Herlaadknop verplaatst HIER)
# STANDAARD GESLOTEN: Minder vaak nodig
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
    
    # 💥 Sectie 2 & 3: Grafiek Opties (Variabelen & Tijd) - GECONSOLIDEERD
    with st.sidebar.expander("📈 Grafiek Opties (Variabele & Tijd)", expanded=True):
        
        # --- 2. Grafiek Variabelen ---
        st.markdown("**1. Grafiek Variabele**") # Compacte titel
        
        plot_options = [COL_DISPLAY_MAP[col] for col in NUMERIC_COLS if col in df_combined.columns]
        
        default_selection = []
        temp_display_name = COL_DISPLAY_MAP.get('temp')
        
        if temp_display_name in plot_options:
            default_selection = [temp_display_name]
            
        if default_selection:
             default_index = plot_options.index(default_selection[0])
        else:
             default_index = 0
             
        selected_variable_display = st.selectbox(
            "Kies de variabele voor de grafiek:",
            options=plot_options,
            index=default_index, 
            key='variable_select_' + '_'.join(selected_station_ids),
            # label_visibility="collapsed" # Houden we voor de leesbaarheid als 'Kies de variabele'
        )
        
        selected_variables = [DISPLAY_TO_COL_MAP[selected_variable_display]]
        
        # --- Scheiding tussen Variabele en Tijd ---
        st.markdown("---")
        
        # --- 3. Tijdsselectie (Live/Recent Data) ---
        st.markdown("**2. Tijdsbereik**") # Compacte titel
        
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
            index=0,
            on_change=lambda: set_active_tab(0) # Spring naar Grafiek tab (index 0)
        )
        
        # <<< NIEUWE POSITIE: Toon Datapunten (NA Tijdselectie) >>>
        # st.markdown("---") # VERWIJDERD VOOR COMPACTHEID
        show_markers = st.checkbox(
            "Toon Datapunten (Markers)", 
            key="show_markers",
            on_change=lambda: set_active_tab(0)
        )
        # <<< EINDE NIEUWE POSITIE >>>

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
            
            # De date_input zelf is al compact.
            custom_date_range = st.date_input(
                "Kies start- en einddatum:",
                value=[max_date_available, max_date_available], 
                min_value=min_date_available,
                max_value=max_date_available,
                key='custom_date_selector_tab12',
                on_change=lambda: set_active_tab(0) # Spring naar Grafiek tab (index 0)
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
    # STANDAARD GESLOTEN: Minder essentiële filtering
    with st.sidebar.expander("❄️ Historische Zoekfilters", expanded=False):
        
        st.markdown(f"**1. Zoekperiode**")
        period_options = ["Onbeperkt (Volledige Database)", "Selecteer Jaar", "Selecteer Maand", "Aangepaste Datums"]
        # TOEVOEGEN: on_change om naar Tab 3 te schakelen
        period_type = st.selectbox(
            "1. Zoekperiode (Historisch)", 
            period_options, 
            key="period_select_hist",
            on_change=lambda: set_active_tab(2) 
        )
        
        df_filtered_time = df_daily_summary.copy()
        
        min_date_hist = df_daily_summary.index.min().date()
        max_date_hist = df_daily_summary.index.max().date()
        
        if period_type == "Selecteer Jaar":
            available_years_hist = sorted(df_daily_summary.index.year.unique())
            st.selectbox(
                "Kies Jaar:", 
                available_years_hist, 
                key="hist_year", 
                on_change=lambda: set_active_tab(2)
            )
            df_filtered_time = df_daily_summary[df_daily_summary.index.year == st.session_state.hist_year]
        
        elif period_type == "Selecteer Maand":
            df_temp_filter = df_daily_summary.copy()
            df_temp_filter['JaarMaand'] = df_temp_filter.index.to_period('M').astype(str)
            available_periods = sorted(df_temp_filter['JaarMaand'].unique(), reverse=True)
            
            st.selectbox(
                "Kies Jaar en Maand (YYYY-MM):", 
                available_periods, 
                key="hist_year_month",
                on_change=lambda: set_active_tab(2)
            )
            selected_period_str = st.session_state.hist_year_month
            df_filtered_time = df_temp_filter[df_temp_filter['JaarMaand'] == selected_period_str].drop(columns=['JaarMaand'])

        elif period_type == "Aangepaste Datums":
            st.date_input(
                "Kies Start- en Einddatum:",
                value=(min_date_hist, max_date_hist),
                min_value=min_date_hist,
                max_value=max_date_hist,
                key="hist_dates",
                on_change=lambda: set_active_tab(2)
            )
            date_range_hist = st.session_state.hist_dates
            
            if len(date_range_hist) == 2:
                df_filtered_time = df_daily_summary.loc[str(date_range_hist[0]):str(date_range_hist[1])]

        st.markdown("---")
        st.markdown(f"**2. Filtertype & Drempel**")

        # TOEVOEGEN: on_change om naar Tab 3 te schakelen
        filter_mode = st.radio(
            "Zoeken op:",
            ["Losse Dagen", "Aaneengesloten Periode", "Hellmann Getal Berekenen"],
            key="filter_mode",
            on_change=lambda: set_active_tab(2) 
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
            # TOEVOEGEN: on_change om naar Tab 3 te schakelen
            min_consecutive_days = st.number_input(
                "Min. aaneengesloten dagen:",
                min_value=2,
                value=3,
                step=1,
                key="min_days_period",
                on_change=lambda: set_active_tab(2) 
            )
            st.markdown("---")
            st.markdown(f"**4. Temperatuurfilter**")
            temp_type_options = ["Max Temp (Temp_High_C)", "Min Temp (Temp_Low_C)", "Gemiddelde Temp (Temp_Avg_C)"]
            # TOEVOEGEN: on_change om naar Tab 3 te schakelen
            temp_type = st.selectbox(
                "Meetwaarde:", 
                temp_type_options, 
                index=2, 
                key="temp_type_period",
                on_change=lambda: set_active_tab(2) 
            )
            temp_column = temp_type.split(" (")[1][:-1] if " (" in temp_type else 'Temp_Avg_C'
            
            # TOEVOEGEN: on_change om naar Tab 3 te schakelen
            comparison = st.radio(
                "Vergelijking:", 
                ["Hoger dan (>=)", "Lager dan (<=)"], 
                key="comparison_period",
                on_change=lambda: set_active_tab(2)
            ) 
            
            # TOEVOEGEN: on_change om naar Tab 3 te schakelen
            temp_threshold = st.number_input(
                "Temperatuur (°C):", 
                value=15.0, 
                step=0.5, 
                key="temp_threshold_period",
                on_change=lambda: set_active_tab(2)
            )

        else: # Losse Dagen
            st.markdown("---")
            st.markdown(f"**4. Temperatuurfilter**")
            temp_type_options = ["Max Temp (Temp_High_C)", "Min Temp (Temp_Low_C)", "Gemiddelde Temp (Temp_Avg_C)"]
            # TOEVOEGEN: on_change om naar Tab 3 te schakelen
            temp_type = st.selectbox(
                "Meetwaarde:", 
                temp_type_options, 
                index=2, 
                key="temp_type_days",
                on_change=lambda: set_active_tab(2)
            )
            temp_column = temp_type.split(" (")[1][:-1] if " (" in temp_type else 'Temp_Avg_C'
            
            # TOEVOEGEN: on_change om naar Tab 3 te schakelen
            comparison = st.radio(
                "Vergelijking:", 
                ["Hoger dan (>=)", "Lager dan (<=)"], 
                key="comparison_days",
                on_change=lambda: set_active_tab(2)
            )
            
            # TOEVOEGEN: on_change om naar Tab 3 te schakelen
            temp_threshold = st.number_input(
                "Temperatuur (°C):", 
                value=15.0, 
                step=0.5, 
                key="temp_threshold_days",
                on_change=lambda: set_active_tab(2)
            )
            
    # 💥 Sectie 5: Maand/Jaar Filters (AANGEPAST: onnodige optie verwijderd)
    # STANDAARD GESLOTEN: Minder essentiële filtering
    with st.sidebar.expander("⚙️ Maand/Jaar Filters", expanded=False):
        # st.header("Analyse Periode") # VERWIJDERD VOOR COMPACTHEID

        # TOEVOEGEN: on_change om naar Tab 4 te schakelen
        analysis_type = st.radio(
            "Kies Analyse Niveau:",
            ["Maand", "Jaar"],
            key="analysis_level_new",
            on_change=lambda: set_active_tab(3)
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
                titel_periode = "Jaar/Maand"
            else: # Jaar
                df_analysis_selector['Jaar'] = df_analysis_selector.index.year
                # Unieke periodes over alle geselecteerde stations
                available_periods = sorted(df_analysis_selector['Jaar'].unique(), reverse=True)
                titel_periode = "Jaar"

            # Check of de key al bestaat en of de selectie in de lijst zit, anders default naar de meest recente
            default_index = 0
            
            # TOEVOEGEN: on_change om naar Tab 4 te schakelen
            selected_period_str = st.selectbox(
                f"Kies de Analyse {titel_periode}:", 
                available_periods, 
                index=default_index,
                key="analysis_period_select_new",
                on_change=lambda: set_active_tab(3)
            )
            
            if analysis_type == "Maand":
                 df_analysis_selector = df_analysis_selector[df_analysis_selector['JaarMaand'] == selected_period_str]
            else: # Jaar
                 df_analysis_selector = df_analysis_selector[df_analysis_selector['Jaar'] == selected_period_str]
            
            # Verwijder de tijdelijke kolommen 
            df_analysis_selector = df_analysis_selector.drop(columns=[c for c in ['JaarMaand', 'Jaar'] if c in df_analysis_selector.columns])
        
# --- Hoofdsectie met Tabs (AANGEPAST VOOR PROGRAMMATISCHE CONTROLE) ---
if df_combined.empty:
    st.warning("Geen weerdata geladen. Selecteer stations in de zijbalk.")
else:
    
    tab_titles_full = ["📈 Grafiek", "📊 Ruwe Data", "🔍 Historische Zoeker", "⭐ Maand/Jaar Analyse", "🏆 Extremen"]
    
    # Hulpfunctie om de actieve index in Session State bij te werken bij handmatige klik op de selectbox
    def update_active_index_on_manual_select():
         # De key 'tab_controller_select' bevat de gekozen titel
         selected_title = st.session_state.tab_controller_select
         try:
             # Update de programmatische index op basis van de handmatige selectie
             st.session_state.active_tab_index = tab_titles_full.index(selected_title)
         except ValueError:
             # Valback naar 0 als er een fout optreedt
             st.session_state.active_tab_index = 0

    # Bepaal de standaardindex op basis van de programmatische instelling
    default_index = st.session_state.active_tab_index

    # Hoofdnavigatie (vervangt st.tabs)
    selected_tab_title = st.selectbox(
        "Navigatie",
        options=tab_titles_full,
        # De index wordt gecontroleerd door de sidebar callbacks via de session state
        index=default_index, 
        key='tab_controller_select',
        on_change=update_active_index_on_manual_select,
        label_visibility='collapsed' # Verberg de label 'Navigatie' voor een compacter uiterlijk
    )
    
    active_tab_index = st.session_state.active_tab_index
    
    # --- Content Rendering op basis van de Actieve Index ---
    
    # We gebruiken een eenvoudige if/elif/else om de inhoud van de JUISTE tab te tonen
    
    if active_tab_index == 0: # --- Tab 1: Grafiek ---
        
        st.header("Grafiek Weergave")
        st.markdown(f"**Periode:** {date_range_display_start} tot {date_range_display_end} (Tijdzone: {TARGET_TIMEZONE})")
        st.markdown(f"**Gevisualiseerde Variabele:** **{selected_variable_display}**")
        
        if not filtered_df.empty:
            
            # --- Nieuwe Samenvattingssectie: Kernwaarden (GEBRUIK ST.METRIC) ---
            st.markdown("---")
            st.subheader("📊 Kernwaarden van de Geselecteerde Periode")
            
            plot_col = selected_variables[0] 
            y_axis_title = selected_variable_display
            
            # Haal de eenheid op met de nieuwe functie (nu inclusief °C voor temperatuur)
            unit = get_unit_from_display_name(y_axis_title, plot_col)

            # Functie om de waarden te formatteren (ZONDER tijdstip)
            def format_value_metric(val):
                if pd.isna(val):
                    return "N/A"
                # Gebruik altijd 1 decimaal voor consistentie in metrics
                return f"{val:.1f} {unit}"
            
            # Groeperen en tonen per station
            for station_name, group in filtered_df.groupby('Station Naam'):
                
                st.markdown(f"##### 📌 Station: **{station_name}**")

                # 1. Berekening van de kernwaarden
                max_val = group[plot_col].max()
                avg_val = group[plot_col].mean()
                min_val = group[plot_col].min()

                last_row = group.sort_values('Timestamp_Local', ascending=False).iloc[0]
                last_val = last_row[plot_col]
                last_time = last_row['Timestamp_Local'].strftime('%H:%M') 
                
                # 2. Bepaal de delta voor de 'Laatste Waarde' (vergelijking met Gemiddelde)
                last_delta_val_str = None
                if not pd.isna(last_val) and not pd.isna(avg_val):
                     delta_raw = last_val - avg_val
                     
                     # Formatteer de delta string (met eenheid en 1 decimaal)
                     if delta_raw > 0:
                         last_delta_val_str = f"+{delta_raw:.1f} {unit}"
                     elif delta_raw < 0:
                          last_delta_val_str = f"{delta_raw:.1f} {unit}"
                     else:
                          last_delta_val_str = "0.0 " + unit

                
                # 3. Toon de metrics in kolommen
                col_last, col_min, col_avg, col_max = st.columns(4)

                with col_last:
                    st.metric(
                        label=f"Laatste ({last_time})", 
                        value=format_value_metric(last_val),
                        delta=f"vs. Gem: {last_delta_val_str}" if last_delta_val_str else None,
                        delta_color="normal" # Standaard kleur om +/- te tonen
                    )
                with col_min:
                    # Zoek de tijd voor de tooltip
                    min_row = group[group[plot_col] == min_val].sort_values('Timestamp_Local', ascending=False).iloc[0]
                    min_time = min_row['Timestamp_Local'].strftime('%H:%M')
                    st.metric(
                        label=f"Minimale ({min_time})", 
                        value=format_value_metric(min_val),
                        # Geen delta hier
                    )
                with col_avg:
                    st.metric(
                        label="Gemiddelde", 
                        value=format_value_metric(avg_val),
                        # Geen delta hier
                    )
                with col_max:
                    # Zoek de tijd voor de tooltip
                    max_row = group[group[plot_col] == max_val].sort_values('Timestamp_Local', ascending=False).iloc[0]
                    max_time = max_row['Timestamp_Local'].strftime('%H:%M')
                    st.metric(
                        label=f"Maximale ({max_time})", 
                        value=format_value_metric(max_val),
                        # Geen delta hier
                    )
                st.markdown("---") 
            # --- Einde st.metric Sectie ---
            
            st.markdown("---") 
            
            plot_col = selected_variables[0] 
            y_axis_title = selected_variable_display
            
            # --- SUGGESTIE A: Bepaal trace mode ---
            trace_mode = 'lines+markers' if st.session_state.get('show_markers') else 'lines'
            # ---------------------------------------

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

            # --- SUGGESTIE B: Custom Tooltip Template & Hovermode ---
            # Dit template wordt gebruikt in combinatie met hovermode="x unified"
            trace_hovertemplate_tab1 = (
                f"<b>{y_axis_title}:</b> %{{y:.1f}} {unit}<br>" +
                "<b>Station:</b> %{full_data.name}<extra></extra>" # <extra></extra> verwijdert de standaard 'trace' naam
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

            # Pas de hoverformat aan voor de X-as (Datum/Tijd)
            fig.update_xaxes(
                 title=f'Tijdstip ({TARGET_TIMEZONE})',
                 hoverformat="%d-%m-%Y %H:%M",
                 tickformat="%H:%M\n%d-%m" # Voor leesbare as labels
            )
            # -------------------------------------------------------------

            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Geen data gevonden voor het geselecteerde tijdsbereik en station(s).")

    elif active_tab_index == 1: # --- Tab 2: Ruwe Data ---
        
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


    elif active_tab_index == 2: # --- Tab 3: Historische Zoeker ---
        
        st.header("Historische Zoeker (Dagelijkse Samenvatting)")
        
        if df_daily_summary.empty:
            st.warning("Geen dagelijkse samenvatting beschikbaar. Laad eerst data.")
        else:
            
            # 1. Definieer de filter logica op basis van de sidebar instellingen
            filtered_data = df_filtered_time.copy()

            if filter_mode == "Hellmann Getal Berekenen":
                
                # --- VISUELE OPSCHONING 1: Subheader met emoji & beknoptere info box ---
                st.subheader(f"❄️ Hellmann Getal Berekening")
                start_date = df_filtered_time.index.min().strftime('%d-%m-%Y')
                end_date = df_filtered_time.index.max().strftime('%d-%m-%Y')
                
                st.info(f"**Analyse Periode:** {period_type} van **{start_date}** tot **{end_date}**.")
                # ---------------------------------------------------------------------

                # Filter voor dagen met Temp_Avg_C <= 0.0
                df_hellmann_days = filtered_data[filtered_data['Temp_Avg_C'] <= 0.0].copy()
                
                if df_hellmann_days.empty:
                    st.success("Er zijn geen dagen gevonden met een Gemiddelde Dagtemperatuur van 0.0 °C of lager in deze periode.")
                else:
                    # Bereken het Hellmann Getal (absolute som van de negatieve gemiddelde temperaturen)
                    hellmann_results = df_hellmann_days.groupby('Station Naam').agg(
                        Aantal_Dagen_Koude=('Temp_Avg_C', 'size'),
                        Hellmann_Getal=('Temp_Avg_C', lambda x: x[x <= 0].abs().sum().round(1))
                    ).reset_index()
                    
                    # Formatteer voor weergave
                    hellmann_results['Hellmann Getal'] = hellmann_results['Hellmann_Getal'].astype(str) + " °C"
                    hellmann_results = hellmann_results.rename(columns={
                        'Aantal_Dagen_Koude': 'Aantal Dagen (≤ 0.0 °C)',
                    }).set_index('Station Naam').drop(columns=['Hellmann_Getal'])
                    
                    # --- GEWENSTE WIJZIGING 2: Kolomvolgorde Hellmann ---
                    hellmann_results = hellmann_results[['Hellmann Getal', 'Aantal Dagen (≤ 0.0 °C)']]
                    # ---------------------------------------------------
                    
                    st.success("Hellmann Getal is de absolute som van de Gemiddelde Dagtemperatuur (alleen als de temperatuur $\\le 0.0$ °C is).")
                    st.dataframe(hellmann_results, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Gevonden Dagen ($\le 0.0$ °C Gem. Temp)")
                    
                    # Toon de dagen zelf
                    df_hellmann_days_display = df_hellmann_days[['Station Naam', 'Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C']].reset_index()
                    df_hellmann_days_display = df_hellmann_days_display.rename(columns={'Date': 'Datum', 'Temp_High_C': 'Max Temp', 'Temp_Low_C': 'Min Temp', 'Temp_Avg_C': 'Gem Temp'})
                    
                    # Formatteer temperaturen
                    for col in ['Max Temp', 'Min Temp', 'Gem Temp']:
                        df_hellmann_days_display[col] = df_hellmann_days_display[col].map(safe_format_temp)
                        
                    df_hellmann_days_display = df_hellmann_days_display.sort_values(['Station Naam', 'Datum'], ascending=[True, False]).set_index('Datum')
                    
                    for station, df_group in df_hellmann_days_display.groupby('Station Naam'):
                        st.markdown(f"##### 📌 Station: **{station}** ({len(df_group)} dagen)")
                        st.dataframe(df_group.drop(columns=['Station Naam']), use_container_width=True)
                        st.markdown("---")


            elif filter_mode == "Aaneengesloten Periode":
                
                # Haal display-waarden op uit session state
                display_temp_type = st.session_state.get('temp_type_period', 'Gemiddelde Temp (Temp_Avg_C)').split(" (")[0]
                comparison_char = "≥" if comparison == "Hoger dan (>=)" else "≤"
                start_date = df_filtered_time.index.min().strftime('%d-%m-%Y')
                end_date = df_filtered_time.index.max().strftime('%d-%m-%Y')
                
                # --- VISUELE OPSCHONING 1: Subheader met emoji & beknoptere info box ---
                st.subheader(f"🔥 Zoekresultaten: Aaneengesloten Periodes van **{min_consecutive_days}**+ dagen")
                
                st.info(f"""
                **Criteria:** {min_consecutive_days}+ aaneengesloten dagen waarbij **{display_temp_type}** {comparison_char} **{temp_threshold}°C**
                
                📆 **Geselecteerde Periode:** {start_date} tot {end_date}
                """)
                # ---------------------------------------------------------------------

                # Filter de data (boolean mask)
                if comparison == "Hoger dan (>=)":
                    df_filtered_period = filtered_data[filtered_data[temp_column] >= temp_threshold]
                else:
                    df_filtered_period = filtered_data[filtered_data[temp_column] <= temp_threshold]

                # Vind aaneengesloten periodes
                periods_df, total_periods = find_consecutive_periods(df_filtered_period, min_consecutive_days, temp_column)
                
                if total_periods > 0:
                    st.success(f"✅ **{total_periods}** aaneengesloten periodes van {min_consecutive_days} dagen of meer gevonden.")
                    
                    # Hernoem kolommen voor weergave
                    periods_df_display = periods_df.rename(columns={'Duur': 'Duur (Dagen)', 'Gemiddelde_Temp_Periode': f'Gem. {temp_column}'})
                    
                    st.dataframe(periods_df_display.set_index('Station Naam'), use_container_width=True)
                    
                else:
                    st.warning("Geen aaneengesloten periodes gevonden die aan de criteria voldoen.")

            else: # Losse Dagen
                
                # Haal display-waarden op uit session state
                display_temp_type = st.session_state.get('temp_type_days', 'Gemiddelde Temp (Temp_Avg_C)').split(" (")[0]
                comparison_char = "≥" if comparison == "Hoger dan (>=)" else "≤"
                start_date = df_filtered_time.index.min().strftime('%d-%m-%Y')
                end_date = df_filtered_time.index.max().strftime('%d-%m-%Y')
                
                # --- VISUELE OPSCHONING 1: Subheader met emoji & beknoptere info box ---
                st.subheader(f"🔍 Zoekresultaten: Losse Dagen")
                
                st.info(f"""
                **Criteria:** Dagen waarbij **{display_temp_type}** {comparison_char} **{temp_threshold}°C**
                
                📆 **Geselecteerde Periode:** {start_date} tot {end_date}
                """)
                # ---------------------------------------------------------------------
                
                # Filter de data (boolean mask)
                if comparison == "Hoger dan (>=)":
                    df_filtered_days = filtered_data[filtered_data[temp_column] >= temp_threshold].copy()
                else:
                    df_filtered_days = filtered_data[filtered_data[temp_column] <= temp_threshold].copy()
                
                total_days = len(df_filtered_days)
                
                if total_days > 0:
                    st.success(f"✅ **{total_days}** dagen gevonden die aan de criteria voldoen.")
                    
                    # Selecteer en hernoem kolommen
                    df_filtered_days_display = df_filtered_days[['Station Naam', 'Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C']].reset_index()
                    df_filtered_days_display = df_filtered_days_display.rename(columns={'Date': 'Datum', 'Temp_High_C': 'Max Temp', 'Temp_Low_C': 'Min Temp', 'Temp_Avg_C': 'Gem Temp'})
                    
                    # Formatteer temperaturen
                    for col in ['Max Temp', 'Min Temp', 'Gem Temp']:
                        df_filtered_days_display[col] = df_filtered_days_display[col].map(safe_format_temp)
                        
                    df_filtered_days_display = df_filtered_days_display.sort_values(['Station Naam', 'Datum'], ascending=[True, False]).set_index('Datum')
                    
                    # Toon resultaten per station
                    for station, df_group in df_filtered_days_display.groupby('Station Naam'):
                        st.markdown(f"##### 📌 Station: **{station}** ({len(df_group)} dagen)")
                        st.dataframe(df_group.drop(columns=['Station Naam']), use_container_width=True)
                        st.markdown("---")
                else:
                    st.warning("Geen dagen gevonden die aan de criteria voldoen in deze periode.")


    elif active_tab_index == 3: # --- Tab 4: Maand/Jaar Analyse ---
        
        st.header(f"Maand/Jaar Analyse: Dagelijkse Samenvatting voor **{selected_period_str}**")
        
        if df_analysis_selector.empty:
            st.warning("Geen dagelijkse data gevonden voor de geselecteerde periode of stations.")
        else:
            
            # --- Samenvattende Tabel ---
            st.subheader("Overzicht per Station")
            
            # Bereken samenvatting per station in de geselecteerde periode
            df_summary_stats = df_analysis_selector.groupby('Station Naam').agg(
                Dagen=('Temp_Avg_C', 'size'),
                Max_Temp_Abs=('Temp_High_C', 'max'),
                Min_Temp_Abs=('Temp_Low_C', 'min'),
                Avg_Temp=('Temp_Avg_C', 'mean'),
                Avg_Druk=('Pres_Avg_hPa', 'mean'),
                Avg_Vocht=('Hum_Avg_P', 'mean'),
            )
            
            # Formatteer de resultaten voor weergave
            df_summary_stats_display = df_summary_stats.rename(columns={
                'Dagen': 'Aantal Dagen',
                'Max_Temp_Abs': 'Abs. Max Temp (°C)',
                'Min_Temp_Abs': 'Abs. Min Temp (°C)',
                'Avg_Temp': 'Gem. Temp Periode (°C)',
                'Avg_Druk': 'Gem. Druk (hPa)',
                'Avg_Vocht': 'Gem. Vocht (%)',
            })
            
            # Pas de formatting toe
            df_summary_stats_display['Abs. Max Temp (°C)'] = df_summary_stats_display['Abs. Max Temp (°C)'].map(safe_format_temp)
            df_summary_stats_display['Abs. Min Temp (°C)'] = df_summary_stats_display['Abs. Min Temp (°C)'].map(safe_format_temp)
            df_summary_stats_display['Gem. Temp Periode (°C)'] = df_summary_stats_display['Gem. Temp Periode (°C)'].apply(lambda x: f"{x:.1f} °C")
            df_summary_stats_display['Gem. Druk (hPa)'] = df_summary_stats_display['Gem. Druk (hPa)'].apply(lambda x: f"{x:.1f} hPa")
            df_summary_stats_display['Gem. Vocht (%)'] = df_summary_stats_display['Gem. Vocht (%)'].apply(lambda x: f"{x:.1f} %")

            st.dataframe(df_summary_stats_display, use_container_width=True)

            # --- Dagelijkse Lijngrafiek (TERUGGEPLAATST) ---
            st.markdown("---")
            st.subheader(f"Dagelijkse Min, Max en Gemiddelde Temperatuur")
            
            # Prepare data for plotting (melt the DataFrame)
            df_plot_daily = df_analysis_selector.reset_index().melt(
                id_vars=['Date', 'Station Naam'],
                value_vars=['Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C'],
                var_name='Temperatuur Type',
                value_name='Temperatuur (°C)'
            )
            
            # Create a mapping for display names
            temp_map = {'Temp_High_C': 'Max', 'Temp_Low_C': 'Min', 'Temp_Avg_C': 'Gemiddeld'}
            df_plot_daily['Temperatuur Type'] = df_plot_daily['Temperatuur Type'].map(temp_map)
            
            # Plotly Line Chart
            fig_daily = px.line(
                df_plot_daily,
                x='Date',
                y='Temperatuur (°C)',
                color='Temperatuur Type', # Color by Max/Min/Avg
                line_dash='Station Naam', # Use line dash for different stations
                title=f'Dagelijkse Min, Max en Gemiddelde Temperatuur in {selected_period_str}',
                labels={'Date': 'Datum', 'Temperatuur (°C)': 'Temperatuur (°C)'},
                height=600,
                custom_data=['Station Naam', 'Temperatuur Type'] # Nodig voor custom tooltip
            )
            
            # --- SUGGESTIE B: Custom Tooltip Template (Dagelijkse data) ---
            trace_hovertemplate_tab4 = (
                "<b>Datum:</b> %{x|%d-%m-%Y}<br>" +
                "<b>%{{customdata[1]}} Temp:</b> %{{y:.1f}} °C<br>" + # %{customdata[1]} is Temp Type (Max/Min/Gemiddeld)
                "<b>Station:</b> %{customdata[0]}<extra></extra>"
            )

            fig_daily.update_traces(
                mode='lines',
                hovertemplate=trace_hovertemplate_tab4
            )
            
            fig_daily.update_layout(hovermode="x unified")
            
            # Pas de hoverformat aan voor de X-as (Datum)
            fig_daily.update_xaxes(
                 title='Datum',
                 hoverformat="%d-%m-%Y"
            )
            # -------------------------------------------------------------

            st.plotly_chart(fig_daily, use_container_width=True)

                
    elif active_tab_index == 4: # --- Tab 5: Extremen ---
        
        st.header("Analyse van Historische Extremen")
        
        if df_daily_summary.empty:
            st.warning("Geen dagelijkse samenvatting beschikbaar. Laad eerst data.")
        else:
            
            # Voer de extreme analyse uit op ALLE beschikbare data (df_daily_summary)
            extreme_results_full = find_extreme_days(df_daily_summary)

            tab_high_max, tab_low_min, tab_high_min, tab_low_max, tab_high_avg, tab_low_avg, tab_range = st.tabs([
                "Hoogste Max", "Laagste Min", "Warmste Nacht", "Koudste Dag", "Hoogste Gem", "Laagste Gem", "Grootste Range"
            ])

            # --- Hoogste Max Temp ---
            with tab_high_max:
                display_extreme_results_by_station(
                    extreme_results_full.get('hoogste_max_temp'),
                    "Hoogste Maximum Temperatuur (Top 5 per station) - De Warmste Dagen",
                    "De dagen waarop de temperatuur overdag het hoogst werd gemeten."
                )

            # --- Laagste Min Temp ---
            with tab_low_min:
                display_extreme_results_by_station(
                    extreme_results_full.get('laagste_min_temp'), 
                    "Laagste Minimum Temperatuur (Top 5 per station) - De Koudste Nachten",
                    "De nachten/momenten waarop de temperatuur het diepst zakte."
                )
                
            # --- Hoogste Min Temp (Warmste Nacht) ---
            with tab_high_min:
                display_extreme_results_by_station(
                    extreme_results_full.get('hoogste_min_temp'), 
                    "Hoogste Minimum Temperatuur (Top 5 per station) - De Warmste Nachten",
                    "De nachten waarop de temperatuur het hoogst bleef (koelde het minst af)."
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
