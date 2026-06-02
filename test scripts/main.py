import ast
from math import atan2, cos, radians, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Φόρτωση των δεδομένων
try:
    df_animals = pd.read_csv("animals.csv")
    df_farms = pd.read_csv("farms.csv")
    df_meteo = pd.read_csv("meteo_data.csv")
    df_devices = pd.read_csv("devices.csv")
    df_device_data = pd.read_csv("device_data.csv")
    print("✅ Όλα τα αρχεία φορτώθηκαν επιτυχώς.")
except FileNotFoundError as e:
    print(f"🛑 Σφάλμα φόρτωσης αρχείου: {e}")
    # Διακοπή εκτέλεσης αν δεν βρεθούν τα αρχεία
    exit()

# ——————————————————————————————————————————————————————————————————————
# 2. Προετοιμασία Δεδομένων & Καθαρισμός Στηλών (για επιτυχή σύνδεση)
# ——————————————————————————————————————————————————————————————————————

# Καθαρισμός/Μορφοποίηση των στηλών σύνδεσης (αφαίρεση περιττών κενών)
def clean_cols(df, col_list):
    for col in col_list:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

df_animals = clean_cols(df_animals, ['breed', 'breed_short', 'farm_id_api', 'id_api'])
df_farms = clean_cols(df_farms, ['id_api'])
df_meteo = clean_cols(df_meteo, ['farm_id_api'])
df_devices = clean_cols(df_devices, ['id_animal'])
df_device_data = clean_cols(df_device_data, ['id_api'])


# ——————————————————————————————————————————————————————————————————————
# 3. Βήμα Συσχέτισης A: Ράτσα -> Φάρμα -> Καιρός
# ——————————————————————————————————————————————————————————————————————

# A1. Σύνδεση animals με farms (για να πάρουμε το όνομα της φάρμας)
# Σύνδεση με βάση το 'farm_id_api'
df_merged_1 = pd.merge(
    df_animals[['id_api', 'name', 'breed_short', 'farm_id_api']],
    df_farms[['id_api', 'name']].rename(columns={'name': 'farm_name', 'id_api': 'farm_id_api'}),
    on='farm_id_api',
    how='left'
)
print("✅ Σύνδεση 1 (Animals & Farms) ολοκληρώθηκε.")

# A2. Σύνδεση μετεωρολογικών δεδομένων με τις ράτσες (βάσει farm_id_api)
# Χρειαζόμαστε τα δεδομένα καιρού για κάθε φάρμα
df_merged_2 = pd.merge(
    df_merged_1,
    df_meteo[['farm_id_api', 'station_timedata', 'temperature', 'humidity', 'heat_index']],
    on='farm_id_api',
    how='left'
)
# Μετονομασία της εξωτερικής θερμοκρασίας και ώρας για σαφήνεια
df_merged_2 = df_merged_2.rename(
    columns={
        'temperature': 'meteo_temperature',
        'station_timedata': 'meteo_time'
    }
)
print("✅ Σύνδεση 2 (Meteo Data) ολοκληρώθηκε.")

# ——————————————————————————————————————————————————————————————————————
# 4. Βήμα Συσχέτισης B: Ράτσα -> Δεδομένα Συσκευής (Δείκτες Υγείας)
# ——————————————————————————————————————————————————————————————————————

# B1. Σύνδεση animals με devices (για να πάρουμε το αναγνωριστικό συσκευής id_api)
# Το id_animal στο devices.csv είναι το id_api του ζώου στο animals.csv
df_devices_map = pd.merge(
    df_animals[['id_api', 'name']],
    df_devices[['id_animal', 'id_api']].rename(columns={'id_api': 'device_id_api'}),
    left_on='id_api',
    right_on='id_animal',
    how='inner'
)
df_devices_map = df_devices_map[['id_api', 'device_id_api']].rename(columns={'id_api': 'animal_id_api'})
print("✅ Σύνδεση Animals & Devices ολοκληρώθηκε.")

# B2. Σύνδεση δεδομένων συσκευής με το χάρτη (map) των ζώων
# Σύνδεση με βάση το 'device_id_api' (το id_api του device_data.csv)
df_device_health = pd.merge(
    df_device_data[['id_api', 'created', 'temperature', 'acc_x', 'std_x']],
    df_devices_map,
    left_on='id_api',
    right_on='device_id_api',
    how='inner'
)
# Μετονομασία της θερμοκρασίας ζώου και ώρας για σαφήνεια
df_device_health = df_device_health.rename(
    columns={
        'id_api': 'device_id_api',
        'temperature': 'animal_temp',
        'created': 'animal_time'
    }
)
df_device_health = df_device_health.drop(columns=['device_id_api'])
print("✅ Σύνδεση Device Data ολοκληρώθηκε.")

# ——————————————————————————————————————————————————————————————————————
# 5. Τελική Συσχέτιση (Ενοποίηση Όλων)
# ——————————————————————————————————————————————————————————————————————

# Τελική σύνδεση του πίνακα Meteo (df_merged_2) με τον πίνακα Health (df_device_health)
# Αυτή η σύνδεση είναι η πιο δύσκολη: Πρέπει να γίνει με βάση το ζώο (animal_id_api) ΚΑΙ το χρόνο.
# Πρέπει να βρούμε τα μετεωρολογικά δεδομένα που είναι κοντά στην ώρα των δεδομένων του ζώου.

# Μετατροπή χρονοσειρών
df_merged_2['meteo_time'] = pd.to_datetime(df_merged_2['meteo_time'], utc=True)
df_device_health['animal_time'] = pd.to_datetime(df_device_health['animal_time'], utc=True)

# Θα χρησιμοποιήσουμε την animal_id_api για να κάνουμε merge_asof (time series merge)
# Η μέθοδος 'merge_asof' είναι ιδανική για να ταιριάξει τη θερμοκρασία του ζώου με τον ΠΙΟ ΠΡΟΣΦΑΤΟ
# μετεωρολογικό σταθμό (ή αυτόν που είναι πιο κοντά χρονικά).

# Προετοιμασία για merge_asof: Ταξινομώ κατά ώρα
df_merged_2 = df_merged_2.sort_values('meteo_time')
df_device_health = df_device_health.sort_values('animal_time')

df_final = pd.merge_asof(
    df_device_health,
    df_merged_2,
    left_on='animal_time',
    right_on='meteo_time',
    by='animal_id_api', # Σύνδεση ανά ζώο
    direction='nearest', # Βρίσκει την πιο κοντινή ώρα (πριν ή μετά)
    tolerance=pd.Timedelta('30min') # Εύρος ανοχής 30 λεπτών
)

# Επιλογή βασικών στηλών για το τελικό αποτέλεσμα
final_cols = [
    'animal_id_api', 'name', 'breed_short', 'farm_name',
    'animal_time', 'animal_temp', 'acc_x', 'std_x',
    'meteo_time', 'meteo_temperature', 'humidity', 'heat_index'
]
df_final = df_final[final_cols].dropna(subset=['meteo_temperature', 'animal_temp'])

print("\n🎉 ΤΕΛΙΚΗ ΣΥΣΧΕΤΙΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ! 🎉")
print(f"Ο τελικός πίνακας 'df_final' έχει {len(df_final)} γραμμές.")
print("\nΕνδεικτικές πρώτες γραμμές του συσχετισμένου πίνακα:")
print(df_final.head())