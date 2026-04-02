import pandas as pd
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
input_file = SCRIPT_DIR.parent / 'data_processed' / 'motion_data.csv'
output_file = SCRIPT_DIR / 'motion_path_corrected.kml'

# Reference Anchor (The Real World GPS location of your "Zero")
ref_lat = 52.05451487197917
ref_lon = 4.349311691811875
start_height = 0.0  # Base height above ground in meters

corr_x = -2
corr_y = 0.8
corr_z = 0

R_earth = 6378137.0 

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: {input_file} not found.")
    exit()

def local_to_kml_coords(row):
    dx = row['Pos_x'] + corr_x
    dy = row['Pos_y'] + corr_y
    dz = row['Pos_z'] + corr_z

    dLat_deg = (dy / R_earth) * (180 / math.pi)

    r_at_lat = R_earth * math.cos(math.radians(ref_lat))
    dLon_deg = (dx / r_at_lat) * (180 / math.pi)

    lat = ref_lat + dLat_deg
    lon = ref_lon + dLon_deg

    alt = dz + start_height

    return f"{lon},{lat},{alt}"

coords_list = df.apply(local_to_kml_coords, axis=1).tolist()
coords_str = " ".join(coords_list)

kml_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Motion Data Path (Corrected)</name>
    <Style id="yellowLine">
      <LineStyle>
        <color>7f00ffff</color> <width>4</width>
      </LineStyle>
      <PolyStyle>
        <color>7f00ff00</color>
      </PolyStyle>
    </Style>
    <Placemark>
      <name>Trajectory</name>
      <styleUrl>#yellowLine</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>relativeToGround</altitudeMode>
        <coordinates>
          {coords_str}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""

output_file.write_text(kml_template, encoding='utf-8')

print(f"Success! {output_file} created with corrections: X={corr_x}, Y={corr_y}, Z={corr_z}")